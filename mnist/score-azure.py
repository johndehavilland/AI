from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.model import Model

from azureml.core import Workspace, Run

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()

def create_env():
    myenv = CondaDependencies()
    myenv.add_conda_package("pytorch")
    myenv.add_conda_package("numpy")
    myenv.add_conda_package("torchvision")

    with open("./myenv.yml","w") as f:
        f.write(myenv.serialize_to_string())
    
    return "./myenv.yml"

def create_config():

    aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                                memory_gb=1, 
                                                tags={"data": "MNIST",  "method" : "pytorch"}, 
                                                description='Predict MNIST with pytorch')
    return aciconfig

def deploy(aciconfig, envfile, name, model):
    # configure the image
    image_config = ContainerImage.image_configuration(execution_script="./score-pytorch-test.py", 
                                                    runtime="python", 
                                                    conda_file=envfile,
                                                    dependencies=["./score/"])

    service = Webservice.deploy_from_model(workspace=ws,
                                        name=name,
                                        deployment_config=aciconfig,
                                        models=[model],
                                        image_config=image_config)

    service.wait_for_deployment(show_output=True)

    print(service.scoring_uri)



if __name__ == '__main__':
    name = "pytorch-mnist-svc"
    model=Model(ws, 'pytorch')
    envfile = create_env()
    aci_config = create_config()
    deploy(aci_config, envfile, name, model)