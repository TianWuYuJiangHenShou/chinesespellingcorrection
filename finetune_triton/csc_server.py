# 导入Flask类
from flask import jsonify
from flask import Flask
from flask import request
from eval import *

# 实例化，可视为固定格式
app = Flask(__name__)

# route()方法用于设定路由；类似spring路由配置
@app.route('/helloworld')
def hello_world():
    return 'Hello, World!'

@app.route('/csc_server', methods=['POST','GET'])
def csc_inference():
    query = request.json
    print(query)
    print(type(query['query']))
    all_inputs,all_fusino_preds = inference(query['query'])
    raw,csc = csc_postprepare(all_inputs,all_fusino_preds)

    # print(raw,csc)
    raw = [''.join(line) for line in raw]
    csc = [''.join(line) for line in csc]

    res = {}
    for i in range(len(raw)):
        index = 'NUM_{}'.format(i)
        res[index] = {'原句':raw[i],'纠错后':csc[i]}

    return jsonify({'result':res})

if __name__ == '__main__':
    # app.run(host, port, debug, options)
    # 默认值：host="127.0.0.1", port=5000, debug=False
    app.run(host="0.0.0.0", port=8888)
