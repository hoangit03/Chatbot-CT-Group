import os
import sys
import pika
import json
import time
from dotenv import load_dotenv

load_dotenv()

# Clean imports (không dùng sys.path hack)
from pipeline.engines.docling_engine import run_ocr_pipeline
from app.services.broker_service import cleaning_publisher

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "ocr_task_queue"
MAX_RETRIES = 3  # Số lần thử lại tối đa trước khi bỏ

def callback(ch, method, properties, body):
    task_data = json.loads(body)
    pdf_name = task_data.get("pdf_name")
    retry_count = task_data.get("retry_count", 0)
    
    if pdf_name:
        print(f"\n[Rabbit Worker] RabbitMQ đã ném cho Worker file: {pdf_name} (lần thử {retry_count + 1})")
        
        # Chuyển giao toàn quyền sinh sát file này cho Tầng Lõi Engine Xử Lý
        success = run_ocr_pipeline(pdf_name)
        
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[Rabbit Worker] ✅ Xác nhận hoàn tất file với RabbitMQ.")
            
            # --- Chuyển Tiếp Sang Trạm Cleaning (Làm Sạch) ---
            base_name = os.path.splitext(pdf_name)[0]
            md_file_name = f"{base_name}_docling.md"
            cleaning_publisher.publish_task(md_file_name)
            print(f"[Rabbit Worker] Đã vận chuyển {md_file_name} tới Cleaning Queue.")
        else:
            # ACK trước (xóa khỏi queue gốc) để tránh block queue
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            if retry_count < MAX_RETRIES:
                # Publish lại với retry_count tăng + delay
                retry_delay = (retry_count + 1) * 30  # 30s, 60s, 90s
                print(f"[Rabbit Worker] ⚠️ Thất bại file {pdf_name}. Thử lại lần {retry_count + 2} sau {retry_delay}s...")
                time.sleep(retry_delay)
                
                retry_task = {"pdf_name": pdf_name, "retry_count": retry_count + 1}
                ch.basic_publish(
                    exchange='',
                    routing_key=QUEUE_NAME,
                    body=json.dumps(retry_task),
                    properties=pika.BasicProperties(delivery_mode=2)
                )
            else:
                print(f"[Rabbit Worker] ❌ Loại bỏ file {pdf_name} sau {MAX_RETRIES} lần thử.")
    else:
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    connection = pika.BlockingConnection(pika.ConnectionParameters(
        host=RABBITMQ_HOST, 
        heartbeat=600,       # Heartbeat mỗi 10 phút (tránh timeout)
        blocked_connection_timeout=3600  # Cho phép block tới 1 giờ
    ))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    print('========================================================')
    print(' [*] OCR Worker đã online và đang hút Job từ Queue...')
    print(f' [*] Max retries: {MAX_RETRIES} | Heartbeat: 600s')
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
