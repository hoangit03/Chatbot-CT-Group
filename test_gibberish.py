import sys; sys.path.insert(0, '.')
from app.services.rag_service import _is_gibberish, _is_skip_rag

tests = [
    ('tôi đang thắc mặc về chế độ nghĩ việc như thế nào', False, 'câu hỏi hợp lệ dài'),
    ('chính sách lương thưởng', False, 'câu hỏi bình thường'),
    ('quy trình tuyển dụng như thế nào', False, 'câu hỏi dài hợp lệ'),
    ('asdfgh', True, 'gibberish ngắn'),
    ('gugugu', True, 'ký tự lặp'),
    ('ok', True, 'quá ngắn'),
    ('sadđ', True, 'vô nghĩa'),
    ('bcdfg', True, 'không nguyên âm'),
]

print('=' * 60)
for text, expected, desc in tests:
    result = _is_gibberish(text)
    status = 'PASS' if result == expected else 'FAIL'
    print(f'  [{status}] gibberish={result} expected={expected} | "{text}" ({desc})')
print('=' * 60)

# Test the actual failing query
real = 'tôi đang thắc mặc về chế độ nghĩ việc như thế nào thì sẽ được hưỡng trọn vẹn tiền lương của bản thân vậy bạn ?'
skip = _is_skip_rag(real)
print(f'\nReal query: skip_rag = {skip} (expected: None)')
