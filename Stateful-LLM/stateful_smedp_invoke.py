import boto3, json


class StatefulSMEDPBuilder:
    def __init__(self, endpoint_name):
        self.endpoint_name = endpoint_name

        ## Boto3 API
        self.sm_bt3_client = boto3.client("runtime.sagemaker")

    def start_session(self, extSessID):
        payload = {
                "extSessionID": extSessID,
                "requestType": 'NEW_SESSION'
            }
        response = self.sm_bt3_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json",
                SessionId="NEW_SESSION"
        )
        return response

    def end_session(self, extSessID):
        payload = {
                "extSessionID": extSessID,
                "requestType": 'CLOSE_SESSION'
            }
        response = self.sm_bt3_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json",
                SessionId=extSessID
        )
        return response

    def invoke(self, textPayload, sampling_params=None, extSessID=None):

        if None == sampling_params:
            sampling_params = {"temperature":0.9, "max_new_token":128, "do_sample":True}

        payload = {
                "inputs": textPayload,
                "sampling_params": sampling_params,
                "extSessionID": extSessID,
                "requestType": 'SESSION'
            }
        
        response = self.sm_bt3_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=json.dumps(payload),
                ContentType="application/json",
                SessionId=extSessID
        )

        return response

    def invoke_stream(self, textPayload, sampling_params=None, extSessID=None):
        pass