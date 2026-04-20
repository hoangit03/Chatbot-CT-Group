import os
import sys
import pika
import json
from dotenv import load_dotenv

load_dotenv()

# Clean imports (không dùng sys.path hack)
from pipeline.engines.paddle_engine import run_ocr_pipeline
from app.services.broker_service import cleaning_publisher

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "ocr_task_queue"

def callback(ch, method, properties, body):
    task_data = json.loads(body)
    pdf_name = task_data.get("pdf_name")
    
    if pdf_name:
        print(f"\n[Rabbit Worker] RabbitMQ đã ném cho Worker file: {pdf_name}")
        
        # Chuyển giao toàn quyền sinh sát file này cho Tầng Lõi Engine Xử Lý
        success = run_ocr_pipeline(pdf_name)
        
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[Rabbit Worker] Xác nhận hoàn tất file với RabbitMQ.")
            
            # --- Chuyển Tiếp Sang Trạm Cleaning (Làm Sạch) ---
            base_name = os.path.splitext(pdf_name)[0]
            md_file_name = f"{base_name}_paddle_only.md"
            cleaning_publisher.publish_task(md_file_name)
            print(f"[Rabbit Worker] Đã vận chuyển {md_file_name} tới Cleaning Queue.")
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print(f"[Rabbit Worker] Loại bỏ Task dị tật: {pdf_name}")
    else:
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=RABBITMQ_HOST, 
        heartbeat=0
    ))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    print('========================================================')
    print(' [*] OCR Worker đã online và đang hút Job từ Queue...')
    print('========================================================')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[Worker] Đã ngưng hoạt động.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
