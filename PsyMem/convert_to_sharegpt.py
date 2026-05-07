import json
import argparse

def convert_to_sharegpt_format(input_file, output_file):
    # 读取原始数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for item in data:
        conversations = []
        
        # 处理每条消息
        for msg in item['messages']:
            if msg['role'] == 'system':
                # ShareGPT 格式通常不包含系统消息
                # 可以选择忽略或将其作为对话的一部分
                # 这里我们将其作为第一条人类消息的前缀
                if conversations and conversations[-1]['from'] == 'human':
                    # 如果上一条是人类消息，将系统消息附加到它
                    conversations[-1]['value'] = msg['content'] + "\n\n" + conversations[-1]['value']
                else:
                    # 否则创建一条新的人类消息
                    conversations.append({
                        'from': 'human',
                        'value': msg['content']
                    })
            elif msg['role'] == 'user':
                conversations.append({
                    'from': 'human',
                    'value': msg['content']
                })
            elif msg['role'] == 'assistant':
                conversations.append({
                    'from': 'gpt',
                    'value': msg['content']
                })
        
        # 添加到转换后的数据
        converted_data.append({
            'conversations': conversations
        })
    
    # 写入转换后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！已将数据保存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将心理咨询数据转换为 ShareGPT 格式')
    # 设置默认输入路径为指定的data.json路径
    parser.add_argument('--input', type=str, 
                        default='/home/capios/inter/interactive-agents-main/data/data.json', 
                        help='输入文件路径')
    # 设置默认输出路径为同一data目录下的sharegpt格式文件
    parser.add_argument('--output', type=str, 
                        default='/home/capios/inter/interactive-agents-main/data/data_sharegpt.json', 
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    convert_to_sharegpt_format(args.input, args.output)