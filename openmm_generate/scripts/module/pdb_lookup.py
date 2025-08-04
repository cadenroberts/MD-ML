import requests
import urllib.parse

def get_rcsb_ligand_smiles(comp_id):
    """
    Returns the smiles string corresponding to the given RCSB compound ID

    Parameters:
    comp_id (str): The RCSB compound id to query

    Returns:
    - smiles_string (str): The SMILES string for the compound, or None if not found
    """
    try:
        return get_rcsb_ligand_smiles_exc(comp_id)
    except Exception as e:
        print("Error in RCSB query")
        print(e)
    return None

def get_rcsb_ligand_smiles_exc(comp_id):
    """
    Returns the smiles string corresponding to the given RCSB compound ID.
    Raises and exception if the compound could not be found.

    Parameters:
    comp_id (str): The RCSB compound id to query

    Returns:
    - smiles_string (str): The SMILES string for the compound
    """
    comp_id = str(comp_id)
    if len(comp_id) != 3:
        raise RuntimeError("Invalid comp_id {comp_id}, must be a 3 letter string.")
    
    query_string = "{chem_comp(comp_id:\"" + comp_id + "\"){chem_comp{id,name,formula},rcsb_chem_comp_descriptor{SMILES,SMILES_stereo}}}"
    query_url = "https://data.rcsb.org/graphql?query=" + urllib.parse.quote(query_string)
    # print(query_url)

    response = requests.get(query_url)
    response.raise_for_status()

    return response.json()["data"]["chem_comp"]["rcsb_chem_comp_descriptor"]["SMILES_stereo"]


if __name__ == "__main__":
    # Note: There are actually many equivilent SMILES strings for a given molecule, but the result returned by RCSB should be determanistic
    from pprint import pprint
    print("Query for \"STR\"")
    query_result = get_rcsb_ligand_smiles("STR")
    expected_smiles = "CC(=O)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CCC4=CC(=O)CC[C@]34C)C"
    pprint(query_result)
    print("Match?:", query_result==expected_smiles)
    print("Query for \"VWW\"")
    query_result = get_rcsb_ligand_smiles("VWW")
    expected_smiles = "c1ccc(cc1)CSC[C@@H](C(=O)N[C@H](c2ccccc2)C(=O)O)NC(=O)CC[C@@H](C(=O)O)N"
    pprint(query_result)
    print("Match?:", query_result==expected_smiles)