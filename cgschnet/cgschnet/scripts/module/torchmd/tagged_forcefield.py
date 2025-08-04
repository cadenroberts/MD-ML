import yaml
from torchmd.forcefields.forcefield import ForceField
from torchmd.forcefields.ff_yaml import YamlForcefield
from ..prior import make_key

def check_directional_bonds(prm):
    """Check if this forcefield contains directional bonds, e.g. both (A, B) and (B, A)."""
    bond_keys = [tuple(k[1:-1].split(", ")) for k in prm["bonds"].keys()]
    # A set of the reversed keys, with anagrams removed
    reversed_bond_keys = set([tuple(reversed(k)) for k in bond_keys if k[0] != k[1]])
    return len(set(bond_keys)) != len(set(bond_keys)-reversed_bond_keys)

def check_if_tagged(prm):
    """Check if this forcefield uses bond tags"""
    return any([a.endswith("*") for a in prm["atomtypes"]])

def create(mol, prm_path):
    with open(prm_path, 'r') as file:
        prm = yaml.safe_load(file)
    if (check_if_tagged(prm)):
        return TaggedYamlForcefield(mol, prm_path)
    return ForceField.create(mol, prm_path)

class TaggedYamlForcefield(YamlForcefield):
    def __init__(self, mol, prm):
        super().__init__(mol, prm)
        assert not check_directional_bonds(self.prm), "TaggedYamlForcefield doesn't support directional bonds"

    def get_parameters(self, term, atomtypes):
        # Only consider tags for bonds
        tagged = len(atomtypes) == 2
        atomtypes = make_key(atomtypes, tagged)
        return super().get_parameters(term, atomtypes)