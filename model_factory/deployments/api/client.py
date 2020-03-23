from typing import Any, Dict

import click
import grpc
import yaml

import numerai_pb2
import numerai_pb2_grpc
import utils

LOGGER = utils.get_logger(__name__)


@click.group()
@click.option('-h', '--host', default='localhost',
              help='Server host name', show_default=True)
@click.option('-p', '--port', type=int, default=50052,
              help='Server port number', show_default=True)
@click.option('-s/-i', '--secure/--insecure', default=False,
              help='Use a secure channel or not', show_default=True)
@click.pass_context
def cli(ctx: Any, host: str, port: int, secure: bool):
    """Connect to NumerAIEngineAPI GRPC server"""
    ctx.ensure_object(dict)
    LOGGER.info(f"Using secure channel: {secure}")
    server_addr = f'{host}:{port}'
    channel_context = (
        grpc.insecure_channel(server_addr) if not secure
        else grpc.secure_channel(server_addr, grpc.ssl_channel_credentials())
    )
    ctx.obj['channel_context'] = channel_context


 @cli.command()
 @click.option('-cfg', '--config', type=str, required=True,
    default='deployments/api/training_config.yml', help='Path to training config')
 @click.option('-cmp', '--competition', type=str,
    required=False, help='Name of current competition')
 @click.pass_context
 def train(ctx: Dict, config: str, competition: str):
     """Train and store model in s3 bucket"""
     with ctx.obj['channel_context'] as channel:
         stub = numerai_pb2_grpc.NumerAIEngineAPIStub(channel)
         train_request = numerai_pb2.TrainRequest(config=config, competition=competition)
         response = stub.Train(train_request)
         LOGGER.info(f'Response: {response}')
         return response


 @cli.command()
 @click.option('-cfg', '--config', type=str,
    default='deployments/api/training_config.yml',
    required=False, help='Path to inference config')
 @click.option('-cmp', '--competition', type=str, 
    required=False, help='Name of current competition')
 @click.option('-sub', '--submit', type=bool, 
    required=False, help='Submit predictions to live NumerAI tournamenet')
 @click.pass_context
 def predict(ctx: Dict, config: str, competition: str, submit: bool):
     """Load model from s3 and conduct inference"""
    with ctx.obj['channel_context'] as channel:
        stub = numerai_pb2_grpc.NumerAIEngineAPIStub(channel)
        inference_request = numerai_pb2.InferenceRequest(
            config=config, 
            competition=competition, 
            submit=submit)
        response = stub.Predict(inference_request)
        LOGGER.info(f'Response: {response}')
        return response
