import pickle
import argparse
import numpy as np

RATING_FILE_NAME = dict(
    {
        "book": "BX-Book-Ratings.csv",
        "music": "user_artists.dat",
        "movie": "ratings.dat",
    }
)

SEP = dict({"book": ";", "music": "\t", "movie": "::"})
THRESHOLD = dict({"book": 0, "music": 0, "movie": 0})


def read_item_index_to_entity_id_file():
    file = "data/" + DATASET + "/item_index2entity_id.txt"
    print("reading item index to entity id file: " + file + " ...")
    i = 0
    for line in open(file, encoding="utf-8").readlines():
        item_index = line.strip().split("\t")[0]
        satori_id = line.strip().split("\t")[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    file = "data/" + DATASET + "/" + RATING_FILE_NAME[DATASET]

    print("reading rating file ...")
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()
    user_item_raw_rating = dict()

    for line in open(file, encoding="utf-8").readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == "book":
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        user_item_raw_rating[(user_index_old, item_index)] = rating
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add(item_index)

    print("converting rating file ...")
    save_path = "data/" + DATASET + "/ratings_final"
    writer = open(save_path + ".txt", "w", encoding="utf-8")
    raw_rating_save_path = "data/" + DATASET + "/raw_ratings_final"
    raw_rating_writer = open(raw_rating_save_path + ".txt", "w", encoding="utf-8")
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        for item in pos_item_set:
            writer.write("%d\t%d\t1\n" % (user_index, item))
            rating = int(user_item_raw_rating[(user_index_old, item)])
            raw_rating_writer.write("%d\t%d\t%d\n" % (user_index, item, rating))

        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            unwatched_set -= user_neg_ratings[user_index_old]

        if len(pos_item_set) < len(unwatched_set):
            neg_items = np.random.choice(
                list(unwatched_set), size=len(pos_item_set), replace=False
            )
        else:
            neg_items = list(unwatched_set)
        for item in neg_items:
            writer.write("%d\t%d\t0\n" % (user_index, item))

    writer.close()
    raw_rating_writer.close()

    ratings_np = np.loadtxt(save_path + ".txt", dtype=np.int64)
    np.save(save_path + ".npy", ratings_np)
    print("number of users: %d" % user_cnt)
    print("number of items: %d" % len(item_set))


def convert_kg():
    print("converting kg file ...")
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    save_path = "data/" + DATASET + "/kg_final"
    writer = open(save_path + ".txt", "w", encoding="utf-8")
    lines = open("data/" + DATASET + "/kg.txt", encoding="utf-8").readlines()

    for ind in range(len(lines)):
        array = lines[ind].strip().split("\t")
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write("%d\t%d\t%d\n" % (head, relation, tail))
    writer.close()

    kg_np = np.loadtxt(save_path + ".txt", dtype=np.int64)
    np.save(save_path + ".npy", kg_np)
    print("number of entities (containing items): %d" % entity_cnt)
    print("number of relations: %d" % relation_cnt)


if __name__ == "__main__":
    np.random.seed(2021)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", type=str, default="movie", help="which dataset to preprocess")
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print("done")
