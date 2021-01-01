import torch
from attractor import BertAttractor
import argparse

def average_pt_models(args):
    device = torch.device('cpu')

    model = BertAttractor()

    snapshot_weights = {}
    for snapshot in args.snapshots:
        checkpoint = torch.load(snapshot, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        snapshot_weights[snapshot] = dict(model.named_parameters())

    dict_params = dict(model.named_parameters())
    N = len(snapshot_weights)

    for name in dict_params.keys():
        custom_param = None
        for _, snapshot_params in snapshot_weights.items():
            if custom_param is None:
                custom_param = snapshot_params[name].data
            else:
                custom_param += snapshot_params[name].data
        dict_params[name].data.copy_(custom_param/N)

    model_dict = model.state_dict()
    model_dict.update(dict_params)

    model.load_state_dict(model_dict)
    torch.save({'model_state_dict': model.state_dict()}, args.out_dir + '/' + args.name)

def main():
    parser = argparse.ArgumentParser(description='Average model parameters', add_help=True)
    parser.add_argument('name', help='model name')
    parser.add_argument('out_dir', help='output directory')
    parser.add_argument('snapshots', nargs='+', help='snapshot files')
    args = parser.parse_args()

    print(args)

    # average the model
    average_pt_models(args)

    print('Finished!')

if __name__=='__main__':
    main()
