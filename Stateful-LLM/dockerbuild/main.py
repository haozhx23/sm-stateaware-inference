from sanic import Sanic, text
from sanic.response import json
from sanic import response
import pickle
import numpy as np
from datetime import datetime, timedelta
import datetime as dt
import uuid
import os
import sglang as sgl

# Create an instance of the Sanic app
app = Sanic("sanic-server")

# 全局状态存储
state_store = {}

@app.route('/ping', methods=['GET'])
def ping(request):
    # Check if the classifier was loaded correctly
    health = engine is not None
    status = 200 if health else 404
    return json(body={}, status=status)


# Define an asynchronous route handler
@app.route("/invocations", methods=["POST"])
async def generate(request):
    print("####################################################################")
    print("All headers:", request.headers)
    reqType = request.json.get("requestType")
    extSessID = request.json.get("extSessionID")

    if 'NEW_SESSION' == reqType:
        current_time = datetime.now(dt.timezone.utc)
        future_time = current_time + timedelta(minutes = int(os.environ['SES_TTL_MIN']))
        formatted_time = future_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        response = json({})
        # response.headers["X-Amzn-SageMaker-Session-Id"] = f'{uuid.uuid4()}; Expires={formatted_time}'
        response.headers["X-Amzn-SageMaker-Session-Id"] = f'{extSessID}; Expires={formatted_time}'
        return response
    elif 'CLOSE_SESSION' == reqType:
        response = json({})
        response.headers["X-Amzn-SageMaker-Closed-Session-Id"] = extSessID
        return response
    else:
        request.conn_info.ctx.sid_4_sm = request.json.get("extSessionID")
    print("####################################################################")

    prompt = request.json.get("inputs")
    if not prompt:
        return json({"error": "inputs is required"}, status=400)

    inf_params = request.json.get("parameters")
    if_stream = request.json.get("stream")

    if if_stream:
        result = await engine.async_generate(prompt=prompt, sampling_params=inf_params, stream=True)
        response = await request.respond()
        
        # if newSessFlag:
        #     response.headers["X-Amzn-SageMaker-Session-Id"] = f'{request.conn_info.ctx.sid_4_sm}; Expires={formatted_time}'

        async for chunk in result:
            await response.send(chunk["text"])
        await response.eof()


    # async_generate returns a dict
    result = await engine.async_generate(prompt=prompt, sampling_params=inf_params)
    response = json({"generation": result})

    # if newSessFlag:
    #     response.headers["X-Amzn-SageMaker-Session-Id"] = f'{request.conn_info.ctx.sid_4_sm}; Expires={formatted_time}'

    return response


def run_server():
    # from cowsay import cow
    # cow(f'Hello World')

    from pyfiglet import Figlet
    f = Figlet()
    print(f.renderText("Amazon SageMaker Host"))

    mid_or_path = os.environ['MODEL_ID_OR_S3_PATH']
    if mid_or_path.startswith('s3://'):
        print('down load from s3')
        dest_path = '/tmp/'
        mid_or_path += '*' if mid_or_path.endswith('/') else '/*'
        os.system(f'./s5cmd cp {mid_or_path} {dest_path}')
        mid_or_path = dest_path

    global engine
    engine = sgl.Engine(model_path = mid_or_path,
                        port=8080,
                        context_length=int(os.environ['CONTEXT_LEN']),
                        # enable_torch_compile=True,
                        mem_fraction_static=float(os.environ['MEM_FRAC'])) # Qwen/Qwen2-0.5B-Instruct meta-llama/Llama-3.1-8B-Instruct

    app.run(host="0.0.0.0", port=8080, single_process=True)


if __name__ == "__main__":
    run_server()