import os
import sys
import pika
import json
from dotenv import load_dotenv

load_dotenv()

# Clean imports
from pipeline.engines.embedding_engine import run_embedding_pipeline

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "embedding_task_queue"

def callback(ch, method, properties, body):
    task_data = json.loads(body)
    json_file_name = task_data.get("pdf_name")  # API hien tai van dung key pdf_name làm Payload chính
    
    if json_file_name:
        print(f"\n[Embedding Worker] RabbitMQ đã ném cho Thợ Cày file JSON: {json_file_name}")
        
        # Chuyển giao toàn quyền chuyển hóa Vector cho Core
        success = run_embedding_pipeline(json_file_name)
        
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[Embedding Worker] Đã chôn sâu {json_file_name} vào VectorDB thành công.")
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print(f"[Embedding Worker] Loại bỏ Task lỗi: {json_file_name}")
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
    print(' [*] Embedding Worker online! Cầu nối VectorDB (ChromaDB) Sẵn Sàng...')
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
