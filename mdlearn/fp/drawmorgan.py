'''
This file is slightly modified from rdkit.Chem.Draw
_getMorganEnv is modified to show the whole molecule when drawing Morgan Fingerprint
'''
from rdkit import Chem
from rdkit.Chem.Draw import rdDepictor, rdMolDraw2D, FingerprintEnv


def DrawMorganBit(mol, bitId, bitInfo, whichExample=0, **kwargs):
    atomId, radius = bitInfo[bitId][whichExample]
    return DrawMorganEnv(mol, atomId, radius, **kwargs)


def DrawMorganEnv(mol, atomId, radius, molSize=(150, 150), baseRad=0.3, useSVG=True,
                  aromaticColor=(0.9, 0.9, 0.2), ringColor=(0.8, 0.8, 0.8),
                  centerColor=(0.6, 0.6, 0.9), extraColor=(0.9, 0.9, 0.9), **kwargs):
    menv = _getMorganEnv(mol, atomId, radius, baseRad, aromaticColor, ringColor, centerColor,
                         extraColor, **kwargs)

    # Drawing
    if useSVG:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(molSize[0], molSize[1])

    drawopt = drawer.drawOptions()
    drawopt.continuousHighlight = False

    drawer.DrawMolecule(menv.submol, highlightAtoms=menv.highlightAtoms,
                        highlightAtomColors=menv.atomColors, highlightBonds=menv.highlightBonds,
                        highlightBondColors=menv.bondColors, highlightAtomRadii=menv.highlightRadii,
                        **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _getMorganEnv(mol, atomId, radius, baseRad, aromaticColor, ringColor, centerColor, extraColor,
                  **kwargs):
    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)
    bitPath = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atomId)

    # get the atoms for highlighting
    atomsToUse = set((atomId,))
    for b in bitPath:
        atomsToUse.add(mol.GetBondWithIdx(b).GetBeginAtomIdx())
        atomsToUse.add(mol.GetBondWithIdx(b).GetEndAtomIdx())

    # #  enlarge the environment by one further bond
    # enlargedEnv = set()
    # for atom in atomsToUse:
    #     a = mol.GetAtomWithIdx(atom)
    #     for b in a.GetBonds():
    #         bidx = b.GetIdx()
    #         if bidx not in bitPath:
    #             enlargedEnv.add(bidx)
    # enlargedEnv = list(enlargedEnv)
    # enlargedEnv += bitPath
    enlargedEnv = [b.GetIdx() for b in mol.GetBonds()]

    # set the coordinates of the submol based on the coordinates of the original molecule
    amap = {}
    submol = Chem.PathToSubmol(mol, enlargedEnv, atomMap=amap)
    Chem.FastFindRings(submol)
    conf = Chem.Conformer(submol.GetNumAtoms())
    confOri = mol.GetConformer(0)
    for i1, i2 in amap.items():
        conf.SetAtomPosition(i2, confOri.GetAtomPosition(i1))
    submol.AddConformer(conf)
    envSubmol = []
    for i1, i2 in amap.items():
        for b in bitPath:
            beginAtom = amap[mol.GetBondWithIdx(b).GetBeginAtomIdx()]
            endAtom = amap[mol.GetBondWithIdx(b).GetEndAtomIdx()]
            envSubmol.append(submol.GetBondBetweenAtoms(beginAtom, endAtom).GetIdx())

    # color all atoms of the submol in gray which are not part of the bit
    # highlight atoms which are in rings
    atomcolors, bondcolors = {}, {}
    highlightAtoms, highlightBonds = [], []
    highlightRadii = {}
    for aidx in amap.keys():
        if aidx in atomsToUse:
            # color = None
            # if centerColor and aidx == atomId:
            #     color = centerColor
            # elif aromaticColor and mol.GetAtomWithIdx(aidx).GetIsAromatic():
            #     color = aromaticColor
            # elif ringColor and mol.GetAtomWithIdx(aidx).IsInRing():
            #     color = ringColor
            color = ringColor
            if centerColor and aidx == atomId:
                color = centerColor

            if color is not None:
                atomcolors[amap[aidx]] = color
                highlightAtoms.append(amap[aidx])
                highlightRadii[amap[aidx]] = baseRad
        else:
            # drawopt.atomLabels[amap[aidx]] = '*'
            submol.GetAtomWithIdx(amap[aidx]).SetAtomicNum(0)
            submol.GetAtomWithIdx(amap[aidx]).UpdatePropertyCache()
    bidx_to_use = []
    for aidx in atomsToUse:
        a = mol.GetAtomWithIdx(aidx)
        for b in a.GetBonds():
            bidx_to_use.append(b.GetIdx())
    color = extraColor
    for bid in submol.GetBonds():
        bidx = bid.GetIdx()
        if bidx not in envSubmol:
            if bidx not in bidx_to_use:
                bondcolors[bidx] = color
            else:
                bondcolors[bidx] = (0.6, 0.6, 0.6)
        highlightBonds.append(bidx)
    return FingerprintEnv(submol, highlightAtoms, atomcolors, highlightBonds, bondcolors, highlightRadii)
