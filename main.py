from pm4py.objects.petri_net.importer import importer as pnml_import
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
import pm4py.objects.process_tree.utils.generic as generic
from configuration import import_path, file_print
from model_feature import p_length, optionality, parallelism
from tree_feature import make_visible, feature_map
import time
import numpy


if __name__ == '__main__':
    # 1 = bpi2012 / 2 = bpi17
    for dataset in [1, 2]:
        net_path = import_path(dataset)
        net_original_file = net_path
        name = net_original_file[4:11]
        out_path = "output/features" + name + ".csv"

        file_print(out_path, ["name activity","path length", "optionality", "par path length", "parallelism",
                              "strectly loopable", "long loopable"], "w")

        net_modified_file = net_original_file.replace(".pnml", "_visible.pnml")
        with open(net_original_file) as infile:
            with open(net_modified_file, 'w') as outfile:
                outfile.write(make_visible(infile.read()))

        net, initial_marking, final_marking = pnml_import.apply(net_modified_file)
        net_or, im, fm = pnml_import.apply(net_original_file)  # IMPORTANT: be sure to use the original net
        tree = wf_net_converter.apply(net, initial_marking, final_marking)
        tree_2 = generic.fold(tree)

        op = tree_2._get_operator()
        curr_features = (1, 1, 0, 0)
        ris = feature_map(tree_2)

        out = {}
        for name in ris.keys():
            l = name.label
            out[l] = ris[name]

        # feature indices
        id_par = 0
        id_opt = 1
        id_sloop = 2
        id_lloop = 3

        t1 = time.time()
        path_l = p_length(out, net_or, im)
        t2 = time.time()
        t_dist = round(t2 - t1, 2)
        print("Path Length elaboration time: ", t_dist)

        opt = optionality(out, id_opt)
        t3 = time.time()
        t_opt = round(t3 - t2, 2)
        print("Optionality, elaboration time: ", t_opt)

        paral_mod = parallelism(tree_2, net, out, id_par)
        t4 = time.time()
        t_par = round(t4 - t3, 2)
        print("Parallelism, elaboration time: ", t_par)

        features = {}
        print()
        print("Activities features: ")
        for elem in out:
            if 'tau' in elem or "Inv" in elem:
                continue
            m = []
            if elem in path_l:
                m.append(path_l[elem])
            else:
                m.append(0)
            if elem in opt:
                m.append(opt[elem])
            else:
                m.append(1)
            if elem in paral_mod:
                m += paral_mod[elem]
            else:
                m += [0, 1]
            m.append(out[elem][id_sloop])
            m.append(out[elem][id_lloop])

            np_m = numpy.array(m)
            features[elem] = np_m

            print(elem)
            print(m)
            m.insert(0, elem)
            file_print(out_path, m)

        if dataset == 1:
            bpi12 = features.copy()
        elif dataset == 2:
            bpi17 = features.copy()

        t5 = time.time()
        t_tot = round(t5 - t1, 2)
        print("Total time execution: ", t_tot)

    k12 = list(bpi12.keys())
    k17 = list(bpi17.keys())
    k12.sort()
    k17.sort()
    file_print("output/matrix.csv", "Similarity inter-process bpi12 - bpi17", 'w')
    file_print("output/matrix.csv", k12)
    for i in range(0, len(k17)):
        att1 = k17[i]
        matrix = [att1]
        for j in range(0, len(k12)):
            att2 = k12[j]
            d = numpy.linalg.norm(bpi17[att1] - bpi12[att2])
            d = round(d, 4)
            matrix.append(d)
        file_print("output/matrix.csv", matrix)

    file_print("output/matrix.csv", '')
    file_print("output/matrix.csv", "Similarity intra-process bpi12")
    file_print("output/matrix.csv", k12)
    for i in range(0, len(k12)):
        att1 = k12[i]
        matrix = [att1]
        for j in range(0, len(k12)):
            att2 = k12[j]
            d = numpy.linalg.norm(bpi12[att1] - bpi12[att2])
            d = round(d, 4)
            matrix.append(d)
        file_print("output/matrix.csv", matrix)

    file_print("output/matrix.csv", '')
    file_print("output/matrix.csv", "Similarity intra-process bpi17")
    file_print("output/matrix.csv", k17)
    for i in range(0, len(k17)):
        att1 = k17[i]
        matrix = [att1]
        for j in range(0, len(k17)):
            att2 = k17[j]
            d = numpy.linalg.norm(bpi17[att1] - bpi17[att2])
            d = round(d, 4)
            matrix.append(d)
        file_print("output/matrix.csv", matrix)
