import requests
import json
import copy

default_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0",
}

identifier_cache = "../cache/identifier_cache.jsonl" # used to save cache of historical queries 

def identifier_resolver(
    identifier: str,
    out_representation: list | None = None,
) -> dict:
    """
    get different representations of single molecule from national cancer institute web api.
    :param identifier: representation of single molecure, could be one of iupac name, smiles, cas number and so on.
    :param out_representation: different kind of representations that you want, should be subset of ["iupac_name", "smiles", "cas", "names"]
    :return: dict of out_representation, {"iupac_name", "smiles", "cas", "names"}
    """
    representations = ["iupac_name", "smiles", "cas", "names"]
    assert set(representations) >= set(out_representation), f"out_representation must be subset of {representations}"

    full_dict = {"identifier": identifier}
    output_dict = {}

    for representation in representations:
        url = f"https://cactus.nci.nih.gov/chemical/structure/{identifier}/{representation}"
        result = requests.get(url)
        if result.status_code == 200:
            if representation != "names":
                full_dict.update({representation:result.text})
            else:
                full_dict.update({representation:result.text.split("\n")})
        else:
            full_dict.update({representation:None})
    with open(identifier_cache, 'a', encoding="utf8") as wf:
        wf.write(json.dumps(full_dict, ensure_ascii=False)+"\n")
    
    if out_representation:
        for odkey in full_dict.keys():
            if odkey in out_representation:
                output_dict[odkey] = full_dict[odkey]
    else:
        output_dict = copy.deepcopy(full_dict)
    return output_dict
    

if __name__ == "__main__":
    print(identifier_resolver("Sodium chloride", ["iupac_name", "cas"]))
