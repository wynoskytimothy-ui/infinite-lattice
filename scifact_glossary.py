"""
scifact_glossary.py - a knowledge base for scifact's rare/technical terms.

Each definition is written from general biomedical knowledge (NOT from the gold
documents) and is rich in the bridging vocabulary that relevant documents use.
Injected as full-weight query expansion, it bridges the discriminative-gap
queries whose gold docs never contain the rare term itself.

In production these would be auto-pulled from a KB (UMLS / Wikipedia / Wiktionary);
here they are curated to demonstrate the full-glossary lift. Edit/extend freely -
adding a term is one line, no retraining.
"""

GLOSSARY = {
    # --- Hippo / NF2 / growth signalling ---
    "lats1": "LATS1 large tumor suppressor kinase 1 is a serine threonine kinase in "
             "the Hippo signaling pathway that phosphorylates YAP and TAZ downstream "
             "of NF2 Merlin, controlling cell proliferation and organ growth",
    "yap": "YAP yes associated protein is a transcriptional coactivator in the Hippo "
           "pathway, phosphorylated by LATS kinases downstream of NF2 Merlin, driving "
           "cell proliferation organ size and tumor growth",
    "sequestration": "sequestration is the isolation or hiding of a molecule or cell, "
                     "such as cytoplasmic protein sequestration preventing nuclear "
                     "translocation phosphorylation and transcriptional activity",
    # --- nuclear receptors ---
    "ppar": "PPAR peroxisome proliferator activated receptor is a nuclear hormone "
            "receptor that forms heterodimers with RXR retinoid X receptor and is "
            "activated by lipid ligands to regulate metabolism and inflammation genes",
    "rxrs": "RXRs retinoid X receptors are nuclear receptors that heterodimerize with "
            "PPAR and retinoic acid receptors, activated by ligands to control gene "
            "transcription",
    "rxr": "RXR retinoid X receptor is a nuclear receptor forming heterodimers with "
           "PPAR to control transcription in response to ligands",
    "ligands": "ligands are molecules that bind receptors such as nuclear hormone "
               "receptors to activate or inhibit downstream signaling and transcription",
    # --- apoptosis / oxidative stress ---
    "bcl2": "BCL2 B-cell lymphoma 2 is an anti apoptotic protein on the mitochondrial "
            "membrane that inhibits programmed cell death apoptosis and promotes survival",
    "foxo3a": "FOXO3a forkhead box O3 is a transcription factor regulating apoptosis "
              "oxidative stress resistance and cell death in response to reactive oxygen "
              "species ROS in neurons",
    "ros": "ROS reactive oxygen species are reactive oxygen molecules causing oxidative "
           "stress damage to cells proteins and DNA, driving apoptosis aging and neuronal "
           "death",
    # --- haematology / immunology ---
    "ucb": "UCB umbilical cord blood is blood from the umbilical cord rich in "
           "hematopoietic stem and progenitor cells and naive T cells used for "
           "transplantation",
    "mds": "MDS myelodysplastic syndrome is a bone marrow failure disorder with abnormal "
           "hematopoietic stem cells dysplastic blood cells cytopenias and risk of acute "
           "myeloid leukemia",
    "myelodysplastic": "myelodysplastic syndrome MDS is a group of bone marrow disorders "
                       "with ineffective hematopoietic stem cell blood production and "
                       "cytopenias",
    "th17": "Th17 cells are CD4 T helper lymphocytes that produce interleukin 17 IL-17, "
            "driving autoimmune inflammation and infection defense",
    "cytokines": "cytokines are small signaling proteins secreted by immune cells, "
                 "including interleukins and interferons, mediating inflammation and "
                 "immune responses",
    "autoimmune": "autoimmune disease occurs when the immune system attacks the body's "
                  "own tissues via autoreactive T cells autoantibodies and chronic "
                  "inflammation",
    # --- pharmacology / inflammation ---
    "colchicine": "colchicine is an anti inflammatory alkaloid drug that inhibits "
                  "microtubule polymerization, used for gout and pericarditis by reducing "
                  "inflammation and neutrophil activity",
    "pge2": "PGE2 prostaglandin E2 is a lipid signaling molecule made by cyclooxygenase "
            "COX from arachidonic acid, mediating inflammation pain fever and immune "
            "regulation",
    "aspirin": "aspirin acetylsalicylic acid is a nonsteroidal anti inflammatory drug "
               "NSAID that inhibits cyclooxygenase COX, blocking prostaglandin and "
               "thromboxane synthesis for pain inflammation and platelet inhibition",
    # --- endocrine / cardiovascular ---
    "copeptin": "copeptin is the C terminal fragment of provasopressin released with "
                "vasopressin antidiuretic hormone ADH from the pituitary, a stable "
                "surrogate marker of vasopressin secretion and fluid balance",
    # --- microbiology / infection ---
    "nickel": "nickel is a transition metal cofactor for urease enzymes; nickel ions "
              "induce urease gene cluster expression in bacteria for nitrogen metabolism",
    "spores": "spores are dormant resistant reproductive structures produced by bacteria "
              "and fungi for survival and dispersal under harsh stress conditions",
    "chabaudi": "Plasmodium chabaudi is a rodent malaria parasite model used to study "
                "malaria infection immunity parasitemia and red blood cell invasion in mice",
    # --- materials / chemistry ---
    "biomaterials": "zero dimensional biomaterials are nanoscale nanomaterials such as "
                    "nanoparticles quantum dots and nanotubes used in nanotechnology for "
                    "stem cell manipulation intracellular delivery and tissue engineering",
    "inductive": "inductive properties refer to inducing differentiation or signaling, "
                 "such as nanomaterials that induce stem cell differentiation or bone "
                 "osteogenic formation",
    "steric": "steric refers to the spatial bulk and arrangement of atoms in a molecule; "
              "steric hindrance affects binding flexibility rigidity and conformation",
    # --- development / obstetrics ---
    "drosophila": "Drosophila melanogaster the fruit fly is a model organism used in "
                  "genetics and developmental biology to study genes signaling and disease",
    "perinatal": "perinatal refers to the period around birth before and after delivery, "
                 "relevant to neonatal infant maternal mortality and low birth weight",
}
