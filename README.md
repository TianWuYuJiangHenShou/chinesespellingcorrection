# chinesespellingcorrection

- paper :PLOME: Pre-training with Misspelled Knowledge for Chinese Spelling Correction
    - https://github.com/liushulinle/PLOME


环境：
- TensorFlow 1.15
- Python3.x


目录结构
- chinesespellingcorrection
    - pre_train_src         
    - finetune_triton    
        - pinyin_data
            - 拼音字典 
        - stroke_data
            - 偏旁字典    
        - datas
            - finance_train.txt
            - finance_train.txt    
                - 保险数据
        - train_eval_tagging.py
            - fine-tune脚本
        - csc_inference.py      
            - 批量测试脚本
        - inference.py          
            - 同上，这人里面只有function
        - flask_server.py       
            - flask服务测试脚本
        - convert_model.py      
            - tf1.x checkpoint转pb
        - export2onnx.sh        
            - pb转onnx的脚本
    - triton                    
        - triton pipeline文件 ，plome.onnx文件已删除
    
