from difflib import SequenceMatcher
import networkx as nx


def roy():
    return "ok"

def text_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

def pair_similarity(script, thress=0.5):
    # print("hk: ", script)
    similar_pairs = []
    for i in range(len(script)):
        for j in range(len(script)):
            if script[i]==script[j]:
                continue
            similarity = text_similarity(script[i]['answer'], script[j]['answer'])
            if similarity >= thress:
                similar_pairs.append({
                            "id1": script[i]['key'], 
                            "id2": script[j]['key'],
                            "similarity": similarity
                        })
    return similar_pairs

def group_pair(similar_pairs):
    G = nx.Graph()
    for pair in similar_pairs:
        id1 = pair['id1']
        id2 = pair['id2']
        weight = pair['similarity']
        G.add_edge(id1, id2, weight=weight)
    
    connected_components = list(nx.connected_components(G))
    
    connected_components_dict = {}
    for idx, group in enumerate(connected_components, 1):
        key = f"group_{idx}"
        connected_components_dict[key] = list(group)
        
    return connected_components_dict
