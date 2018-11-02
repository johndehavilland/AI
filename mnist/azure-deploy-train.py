import numpy as np
import os

from azureml.train.dnn import PyTorch

import azureml
from azureml.core import Workspace, Run
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException

from azureml.train.estimator import Estimator

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()

print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

batchaigpu_cluster_name = "gputraincluster"

def setup_azure_gpu():
    print("Setting up Azure Compute")
    # check core SDK version number
    print("Azure ML SDK Version: ", azureml.core.VERSION)

    try:
        # look for the existing cluster by name
        compute_target = ComputeTarget(workspace=ws, name=batchaigpu_cluster_name)
        if type(compute_target) is BatchAiCompute:
            print('found compute target {}, just use it.'.format(batchaigpu_cluster_name))
        else:
            print('{} exists but it is not a Batch AI cluster. Please choose a different name.'.format(batchaigpu_cluster_name))
        return compute_target
    except ComputeTargetException:
        print('creating a new compute target...')
        compute_config = BatchAiCompute.provisioning_configuration(vm_size="STANDARD_NC6", # GPU
                                                                    #vm_priority='lowpriority', # optional
                                                                    autoscale_enabled=True,
                                                                    cluster_min_nodes=1, 
                                                                    cluster_max_nodes=1)

        # create the cluster
        compute_target = ComputeTarget.create(ws, batchaigpu_cluster_name, compute_config)

        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it uses the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

        # Use the 'status' property to get a detailed status for the current cluster. 
        print(compute_target.status.serialize())

        return compute_target

def run_azure_pytorch():

    compute_target = setup_azure_gpu()

    experiment_name = 'pytorch'

    exp = Experiment(workspace=ws, name=experiment_name)

    ds = ws.get_default_datastore()

    print(ds.datastore_type, ds.account_name, ds.container_name)

    # ds.upload(src_dir='./data', target_path='mnist', overwrite=True, show_progress=True)
    
    script_params = {
        '--data_dir': ds
    }

    pt_est = PyTorch(source_directory='./train-scripts',
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='train-pytorch.py',
                    use_gpu=True)

    run = exp.submit(pt_est)
    run
    run.wait_for_completion(show_output=True)
    print(run.get_metrics())
    print(run.get_file_names())

    # register model 
    model = run.register_model(model_name=experiment_name, model_path='outputs/pytorch_model.pt')
    print(model.name, model.id, model.version, sep = '\t')
    compute_target.delete()

if __name__ == '__main__':
    run_azure_pytorch()

