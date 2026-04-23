import os
import sys
import pika
import json
from dotenv import load_dotenv

load_dotenv()

# Clean imports
from pipeline.engines.chunking_engine import run_chunking_pipeline
from app.services.broker_service import embedding_publisher

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
QUEUE_NAME = "chunking_task_queue"

def callback(ch, method, properties, body):
    task_data = json.loads(body)
    md_file_name = task_data.get("pdf_name")  # Biến này mang ý nghĩa là File Name chung. Key đẩy vào vẫn đang là pdf_name
    
    if md_file_name:
        print(f"\n[Chunking Worker] RabbitMQ đã ném cho Worker file MD: {md_file_name}")
        
        # Chuyển giao toàn quyền sinh sát file này cho Tầng Lõi Engine Xử Lý
        success = run_chunking_pipeline(md_file_name)
        
        if success:
            ch.basic_ack(delivery_tag=method.delivery_tag)
            print(f"[Chunking Worker] Trực quan hóa Chunk thành công.")
            
            # --- Chuyển Tiếp Sang Trạm Embedding ---
            base_name = os.path.splitext(md_file_name)[0]
            json_file_name = f"{base_name}_chunks.json"
            embedding_publisher.publish_task(json_file_name)
            print(f"[Chunking Worker] Đã vận chuyển {json_file_name} tới Embedding Queue.")
            
        else:
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            print(f"[Chunking Worker] Loại bỏ Task dị tật: {md_file_name}")
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
    print(' [*] Chunking Worker đã online và đang hút Job từ Queue...')
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
