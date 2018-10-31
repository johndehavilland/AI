
import sys
import azureml.core
from azureml.core import Workspace
from azureml.core import Experiment



def setup():
    subscription_id =''
    resource_group =''
    workspace_name = ''
    print(azureml.core.VERSION)
    try:
        ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
        ws.write_config()
        print('Library configuration succeeded')
    except:
        print('Workspace not found')
    
    return ws

def test_run(expname, ws):
    # create a new experiment
    exp = Experiment(workspace=ws, name=expname)

    # start a run
    run = exp.start_logging()

    # log a number
    run.log('my magic number', 42)

    # log a list (Fibonacci numbers)
    run.log_list('my list', [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]) 

    # finish the run
    run.complete()

    print(run.get_portal_url())

if __name__ == '__main__':
    ws = setup()
    test_run("test_exp", ws)