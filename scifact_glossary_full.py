"""LLM-written definitions of SciFact rare gap terms -- from GENERAL domain knowledge
(what each term means + the vocabulary a relevant document would naturally use), NOT
reverse-engineered from the qrels/gold docs. This is the 'LLM teacher defines the
gaps' lever for zero-shot retrieval: information the corpus lacks, injected as
query expansion. Swap for a live LLM/Wikipedia call to scale to any corpus."""

GLOSSARY = {
    "yap": "yes associated protein transcriptional coactivator hippo signaling pathway "
           "proliferation organ size lats tead phosphorylation",
    "lats1": "large tumor suppressor kinase hippo pathway phosphorylates yap tumor "
             "suppressor nf2 merlin",
    "ppar": "peroxisome proliferator activated receptor nuclear receptor lipid metabolism "
            "adipogenesis glucose insulin gamma",
    "rxrs": "retinoid x receptor nuclear receptor heterodimer ppar rar vitamin retinoic acid",
    "foxo3a": "forkhead box transcription factor apoptosis oxidative stress longevity "
              "autophagy akt pi3k",
    "raptor": "regulatory associated protein mtor mtorc1 complex nutrient sensing growth "
              "signaling",
    "bcl2": "b cell lymphoma anti apoptotic protein apoptosis mitochondria survival cancer",
    "mda5": "melanoma differentiation associated rig like receptor viral rna innate immunity "
            "interferon sensing",
    "card": "caspase recruitment domain signaling apoptosis inflammation innate immune",
    "th17": "t helper cell interleukin il17 autoimmune inflammation cytokine differentiation",
    "copeptin": "c terminal provasopressin vasopressin biomarker surrogate cardiovascular "
                "sepsis stress",
    "myelodysplastic": "myelodysplastic syndrome bone marrow ineffective hematopoiesis "
                       "cytopenia leukemia blast",
    "mds": "myelodysplastic syndrome bone marrow hematopoiesis cytopenia leukemia",
    "chabaudi": "plasmodium malaria parasite rodent mouse infection erythrocyte",
    "pge2": "prostaglandin inflammation cyclooxygenase cox lipid mediator pain fever",
    "aspirin": "acetylsalicylic acid nonsteroidal anti inflammatory cyclooxygenase "
               "antiplatelet cardiovascular",
    "statins": "hmg coa reductase inhibitor cholesterol lipid lowering cardiovascular ldl",
    "colchicine": "alkaloid microtubule polymerization gout anti inflammatory tubulin",
    "tet": "ten eleven translocation dna demethylation hydroxymethylcytosine epigenetic "
           "methylation",
    "epigenome": "epigenetic dna methylation histone modification chromatin gene expression",
    "neurogenesis": "new neuron formation hippocampus neural stem cell brain dentate",
    "biomaterials": "material implant scaffold tissue engineering biocompatible regenerative",
    "sequestration": "sequester isolate bind infected cell microvasculature protein",
    "methionine": "amino acid sulfur methylation essential restriction dietary",
    "ucb": "umbilical cord blood hematopoietic stem cell transplant",
    "perinatal": "birth pregnancy neonatal maternal mortality infant",
    "noncommunicable": "chronic disease cardiovascular cancer diabetes noninfectious",
    "ionizing": "radiation dna damage double strand break",
    "spores": "dormant bacteria endospore fungi germination resistant",
    "homelessness": "homeless housing shelter social determinant health",
    "nickel": "metal allergen carcinogen catalysis",
    "steric": "spatial molecular hindrance conformation atoms",
    "deoxyribonucleic": "dna nucleic acid genetic",
}
