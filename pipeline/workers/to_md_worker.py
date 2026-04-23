import os
import sys
import pika
import json
from dotenv import load_dotenv

load_dotenv()

# Clean imports
from pipeline.engines.to_md_engine import run_to_md_pipeline
from app.services.broker_service import cleaning_publisher

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "to_md_task_queue"

def callback(ch, method, properties, body):
    task_data = json.loads(body)
    file_name = task_data.get("pdf_name")  # API hien tai van dung key pdf_name chua thong nhat nhung gia tri la doc_name
    
    if file_name:
        print(f"\n[ToMD Worker] RabbitMQ ném file: {file_name}")
        success = run_to_md_pipeline(file_name)
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[ToMD Worker] Xác nhận hoàn tất file: {file_name}")
            
            # --- Chuyển Tiếp Sang Trạm Cleaning (Làm Sạch) ---
            base_name = os.path.splitext(file_name)[0]
            md_file_name = f"{base_name}.md"
            cleaning_publisher.publish_task(md_file_name)
            print(f"[ToMD Worker] Đã vận chuyển {md_file_name} tới Cleaning Queue.")
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print(f"[ToMD Worker] Error xử lý: {file_name}")
    else:
        ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, heartbeat=0))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    
    print('========================================================')
    print(' [*] ToMD Worker đang online và hút Job từ Queue...')
    print('========================================================')
    channel.start_consuming()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n[Worker] Đã ngưng hoạt động.')
        sys.exit(0)
