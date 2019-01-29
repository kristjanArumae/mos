import os, shutil
import torch
from torch.autograd import Variable


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, tuple) or isinstance(h, list):
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.detach()


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data[0].size(0) // bsz

    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_x = data[0].narrow(0, 0, nbatch * bsz)
    data_y = data[1].narrow(0, 0, nbatch * bsz)

    # Evenly divide the data across the bsz batches.
    data_y = data_y.view(bsz, -1, args.num_factors).permute(1, 0, 2).contiguous()
    data_x = data_x.view(bsz, -1).permute(1, 0).contiguous()

    print(data_x.size())
    if args.cuda:
        data_x = data_x.cuda()
        data_y = data_y.cuda()
    return data_x, data_y


def get_batch(source, i, args, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(source[0]) - 1 - i)
    data = Variable(source[0][i:i+seq_len])
    # target = Variable(source[i+1:i+1+seq_len].view(-1))z
    target = Variable(source[1][i+1:i+1+seq_len])
    return data, target


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)

    print('Experiment dir : {}'.format(path))
    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(model, optimizer, path, finetune=False):
    if finetune:
        torch.save(model, os.path.join(path, 'finetune_model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'finetune_optimizer.pt'))
    else:
        torch.save(model, os.path.join(path, 'model.pt'))
        torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer.pt'))


def convert_to_base(x_ls, num_symb, num_factors):
    modified_ls = []

    for item in x_ls:
        extended_item = []
        idx = 0

        while item:
            extended_item.append(item % num_symb)
            item = item // num_symb
            idx += 1

        extended_item.reverse()
        padding = [0] * (num_factors - len(extended_item))
        extended_item.extend(padding)

        modified_ls.append(extended_item)

    return modified_ls
