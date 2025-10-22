import pkg_resources
import os
import errno
from pathlib import Path
import pickle

import numpy as np

from collections import defaultdict

DATA_PATH = pkg_resources.resource_filename('twq', 'data/')
#DATA_PATH = DATA_PATH = Path.cwd() / 'src_data'

def prepare_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
            except ValueError:
                continue
        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(DATA_PATH) / name / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


def prepare_P2_dataset(path, name):
    """
    Given a path to a folder containing tab separated files :
    train, test, valid
    In the format:
    (lhs)\t(rel1)\t(ts1)\t(rel2)\t(ts2)\t(rhs)
    Maps each entity and relation to a unique id, creates corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    """
    #files = ['train', 'valid-easy', 'test-easy']
    files = ['train', 'valid-hard', 'test-hard']
    entities, relations, timestamps = set(), set(), set()

    for f in files:
        file_path = os.path.join(path, f)
        with open(file_path, 'r') as to_read:
            for line in to_read.readlines():
                lhs, rel1, ts1, rel2, ts2, rhs = line.strip().split('\t')
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel1)
                relations.add(rel2)
                timestamps.add(ts1)
                timestamps.add(ts2)

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    os.makedirs(os.path.join(DATA_PATH, name), exist_ok=True)

    # Write ent_to_id / rel_to_id / ts_to_id
    for dic, f in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        with open(os.path.join(DATA_PATH, name, f), 'w+') as ff:
            for x, i in dic.items():
                ff.write("{}\t{}\n".format(x, i))

    # Map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        examples = []
        with open(file_path, 'r') as to_read:
            for line in to_read.readlines():
                lhs, rel1, ts1, rel2, ts2, rhs = line.strip().split('\t')
                try:
                    examples.append([
                        entities_to_id[lhs],
                        relations_to_id[rel1],
                        timestamps_to_id[ts1],
                        relations_to_id[rel2],
                        timestamps_to_id[ts2],
                        entities_to_id[rhs]
                    ])
                except ValueError:
                    continue

        with open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb') as out:
            pickle.dump(np.array(examples).astype('uint64'), out)

    print("creating filtering lists")

    # Create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel1, ts1, rel2, ts2, rhs in examples:
            to_skip['lhs'][(rhs,rel1, ts1, rel2, ts2)].add(lhs)  # Reciprocals for rhs
            to_skip['rhs'][(lhs, rel1, ts1,rel2,ts2)].add(rhs)  # Reciprocals for lhs

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    with open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb') as out:
        pickle.dump(to_skip_final, out)

    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel1, ts1, rel2, ts2, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1

    for k, v in counters.items():
        counters[k] = v / np.sum(v)  # Normalize to probabilities

    with open(Path(DATA_PATH) / name / 'probas.pickle', 'wb') as out:
        pickle.dump(counters, out)

def prepare_P3_dataset(path, name):
    """
    Given a path to a folder containing tab separated files:
    train, test, valid
    In the format:
    (lhs)\t(rel1)\t(ts1)\t(rel2)\t(ts2)\t(rel3)\t(ts3)\t(rhs)
    Maps each entity and relation to a unique id, creates corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    """
    files = ['train', 'valid-easy', 'test-easy']
    #files = ['train', 'valid-hard', 'test-hard']
    entities, relations, timestamps = set(), set(), set()

    # Collect entities, relations, and timestamps
    for f in files:
        file_path = os.path.join(path, f)
        with open(file_path, 'r') as to_read:
            for line in to_read.readlines():
                lhs, rel1, ts1, rel2, ts2, rel3, ts3, rhs = line.strip().split('\t')
                entities.add(lhs)
                entities.add(rhs)
                relations.add(rel1)
                relations.add(rel2)
                relations.add(rel3)
                timestamps.add(ts1)
                timestamps.add(ts2)
                timestamps.add(ts3)

    # Create mappings for entities, relations, timestamps
    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    # Create the folder for saving data
    os.makedirs(os.path.join(DATA_PATH, name), exist_ok=True)

    # Write entity, relation, timestamp mappings to files
    for dic, f in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        with open(os.path.join(DATA_PATH, name, f), 'w+') as ff:
            for x, i in dic.items():
                ff.write("{}\t{}\n".format(x, i))

    # Process train/test/valid files
    for f in files:
        file_path = os.path.join(path, f)
        examples = []
        with open(file_path, 'r') as to_read:
            for line in to_read.readlines():
                lhs, rel1, ts1, rel2, ts2, rel3, ts3, rhs = line.strip().split('\t')
                try:
                    examples.append([
                        entities_to_id[lhs],
                        relations_to_id[rel1],
                        timestamps_to_id[ts1],
                        relations_to_id[rel2],
                        timestamps_to_id[ts2],
                        relations_to_id[rel3],
                        timestamps_to_id[ts3],
                        entities_to_id[rhs]
                    ])
                except ValueError:
                    continue

        # Save the processed examples as pickle files
        with open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb') as out:
            pickle.dump(np.array(examples).astype('uint64'), out)

    print("Creating filtering lists")

    # Create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel1, ts1, rel2, ts2, rel3, ts3, rhs in examples:
            to_skip['lhs'][(rhs, rel1, ts1, rel2, ts2, rel3, ts3)].add(lhs)
            to_skip['rhs'][(lhs, rel1, ts1, rel2, ts2, rel3, ts3)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    with open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb') as out:
        pickle.dump(to_skip_final, out)

    # Generate counters for entity occurrences
    examples = pickle.load(open(Path(DATA_PATH) / name / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel1, ts1, rel2, ts2, rel3, ts3, rhs in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1

    # Normalize to probabilities
    for k, v in counters.items():
        counters[k] = v / np.sum(v)  # Normalize to probabilities

    with open(Path(DATA_PATH) / name / 'probas.pickle', 'wb') as out:
        pickle.dump(counters, out)

if __name__ == "__main__":

    datasets = ['ICEWS14','ICEWS05-15']#, 'ICEWS14-1P'
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
    """
    datasets = [ 'ICEWS14-2P']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_P2_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
    
    datasets = ['ICEWS14-3P']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        try:
            prepare_P3_dataset(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), 'src_data', d
                ),
                d
            )
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(e)
                print("File exists. skipping...")
            else:
                raise
    """
