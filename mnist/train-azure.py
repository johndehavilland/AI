import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import urllib.request
from azureml.train.dnn import PyTorch
from utils import load_data


import azureml
from azureml.core import Workspace, Run
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, BatchAiCompute
from azureml.core.compute_target import ComputeTargetException
from sklearn.linear_model import LogisticRegression

from azureml.train.estimator import Estimator

# load workspace configuration from the config.json file in the current folder.
ws = Workspace.from_config()

print(ws.name, ws.location, ws.resource_group, ws.location, sep = '\t')

batchai_cluster_name = "traincluster"

batchaigpu_cluster_name = "gputraincluster"

def setup_azure():
    print("Setting up Azure Compute")
    # check core SDK version number
    print("Azure ML SDK Version: ", azureml.core.VERSION)

    try:
        # look for the existing cluster by name
        compute_target = ComputeTarget(workspace=ws, name=batchai_cluster_name)
        if type(compute_target) is BatchAiCompute:
            print('found compute target {}, just use it.'.format(batchai_cluster_name))
        else:
            print('{} exists but it is not a Batch AI cluster. Please choose a different name.'.format(batchai_cluster_name))
        return compute_target
    except ComputeTargetException:
        print('creating a new compute target...')
        compute_config = BatchAiCompute.provisioning_configuration(vm_size="STANDARD_D2_V2", # small CPU-based VM
                                                                    #vm_priority='lowpriority', # optional
                                                                    autoscale_enabled=True,
                                                                    cluster_min_nodes=1, 
                                                                    cluster_max_nodes=1)

        # create the cluster
        compute_target = ComputeTarget.create(ws, batchai_cluster_name, compute_config)

        # can poll for a minimum number of nodes and for a specific timeout. 
        # if no min node count is provided it uses the scale settings for the cluster
        compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

        # Use the 'status' property to get a detailed status for the current cluster. 
        print(compute_target.status.serialize())

        return compute_target

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

def get_data():
    os.makedirs('./data', exist_ok = True)

    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', filename='./data/train-images.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', filename='./data/train-labels.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', filename='./data/test-images.gz')
    urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', filename='./data/test-labels.gz')

def loaddata():
    get_data()
    # note we also shrink the intensity values (X) from 0-255 to 0-1. This helps the model converge faster.
    X_train = load_data('./data/train-images.gz', False) / 255.0
    y_train = load_data('./data/train-labels.gz', True).reshape(-1)

    X_test = load_data('./data/test-images.gz', False) / 255.0
    y_test = load_data('./data/test-labels.gz', True).reshape(-1)

    # now let's show some randomly chosen images from the traininng set.
    count = 0
    sample_size = 30
    plt.figure(figsize = (16, 6))
    for i in np.random.permutation(X_train.shape[0])[:sample_size]:
        count = count + 1
        plt.subplot(1, sample_size, count)
        plt.axhline('')
        plt.axvline('')
        plt.text(x=10, y=-10, s=y_train[i], fontsize=18)
        plt.imshow(X_train[i].reshape(28, 28), cmap=plt.cm.Greys)
    plt.show()
    return X_train, X_test, y_test, y_train

def run_local(X_train, y_train, X_test, y_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_hat = clf.predict(X_test)
    print(np.average(y_hat == y_test))

def run_azure_sklearn():
    compute_target = setup_azure()

    experiment_name = 'sklearn-mnist'
    exp = Experiment(workspace=ws, name=experiment_name)

    ds = ws.get_default_datastore()
    print(ds.datastore_type, ds.account_name, ds.container_name)

    ds.upload(src_dir='./data', target_path='mnist', overwrite=True, show_progress=True)

    script_params = {
        '--data-folder': ds.as_mount(),
        '--regularization': 0.8
    }

    est = Estimator(source_directory='./mnist/scripts',
                    script_params=script_params,
                    compute_target=compute_target,
                    entry_script='train-scikit.py',
                    conda_packages=['scikit-learn'])
    
    run = exp.submit(config=est)
    run
    run.wait_for_completion(show_output=True)
    print(run.get_metrics())
    print(run.get_file_names())
    # register model 
    model = run.register_model(model_name=experiment_name, model_path='outputs/sklearn_mnist_model.pkl')
    print(model.name, model.id, model.version, sep = '\t')
    compute_target.delete()

def run_azure_pytorch():

    compute_target = setup_azure_gpu()

    experiment_name = 'pytorch-mnist'
    exp = Experiment(workspace=ws, name=experiment_name)

    ds = ws.get_default_datastore()
    print(ds.datastore_type, ds.account_name, ds.container_name)

    ds.upload(src_dir='./data', target_path='mnist', overwrite=True, show_progress=True)
    
    script_params = {
        '--data_dir': ds
    }

    pt_est = PyTorch(source_directory='./mnist/scripts',
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
    model = run.register_model(model_name=experiment_name, model_path='outputs/pytorch_mnist_model.pkl')
    print(model.name, model.id, model.version, sep = '\t')
    #compute_target.delete()

if __name__ == '__main__':
    
    
    #X_train, X_test, y_test,y_train = loaddata()
    
    #run_local(X_train, y_train, X_test, y_test)
    
    #run_azure_sklearn()

    run_azure_pytorch()

