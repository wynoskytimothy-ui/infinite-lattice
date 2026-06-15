"""
nfcorpus_glossary.py - LLM-distilled knowledge for nfcorpus's nutrition/medical
gap terms. Teacher-LLM definitions from general knowledge (not the gold docs),
rich in the bridging vocabulary the relevant documents use.
"""

GLOSSARY = {
    "brca": "BRCA1 and BRCA2 are tumor suppressor genes involved in DNA repair; "
            "inherited mutations greatly increase the risk of breast and ovarian "
            "cancer",
    "bioavailability": "bioavailability is the fraction of a nutrient or drug that "
                       "is absorbed and available for use by the body after "
                       "ingestion, affecting absorption and effectiveness",
    "thiamine": "thiamine vitamin B1 is an essential water soluble vitamin needed "
                "for carbohydrate metabolism and nerve function; deficiency causes "
                "beriberi and neurological problems",
    "quinine": "quinine is an alkaloid drug from cinchona bark used to treat "
               "malaria, also present in tonic water and bitter drinks",
    "lindane": "lindane is an organochlorine pesticide and insecticide used to "
               "treat lice and scabies, a persistent toxic environmental pollutant",
    "lyme": "Lyme disease is a tick borne bacterial infection caused by Borrelia "
            "spread by deer ticks, causing rash fever fatigue and joint pain",
    "flax": "flax flaxseed linseed is a seed rich in omega 3 fatty acids alpha "
            "linolenic acid fiber and lignans, used for cardiovascular cholesterol "
            "and digestive health",
    "chanterelle": "chanterelle is an edible wild mushroom; identifying mushrooms "
                   "matters because some wild mushrooms are poisonous and toxic",
    "prenatal": "prenatal means before birth during pregnancy; prenatal vitamins "
                "with folic acid folate iron support fetal development and prevent "
                "birth defects",
    "hormonal": "hormonal relates to hormones such as estrogen testosterone and "
                "insulin that regulate the body; hormonal contraception and therapy "
                "affect reproductive and metabolic health",
    "vitamins": "vitamins are essential micronutrients such as vitamin A B C D E "
                "and folate required in small amounts from diet for metabolism "
                "growth and health",
    "pumpkin": "pumpkin and pumpkin seeds are a squash rich in beta carotene "
               "vitamin A fiber and antioxidants",
    "subsidies": "agricultural subsidies are government financial support to "
                 "farmers for crops such as corn and soy, influencing food prices "
                 "and the food supply",
    "physicians": "physicians are medical doctors who diagnose and treat patients "
                  "and conduct clinical health studies and trials",
    "autophagy": "cellular self degradation recycling organelle lysosome mtor fasting "
                 "caloric restriction longevity cancer",
    "carnitine": "amino acid fatty acid transport mitochondria tmao trimethylamine red "
                 "meat carnivore heart",
    "turmeric": "curcumin spice curcuma anti inflammatory antioxidant polyphenol",
    "curcumin": "turmeric curcuma anti inflammatory polyphenol antioxidant",
    "psoriasis": "autoimmune skin disease plaque inflammation scaly immune",
    "fenugreek": "herb spice seed blood sugar glucose lactation testosterone",
    "hernia": "hiatal inguinal protrusion organ abdominal wall reflux",
    "uterine": "uterus endometrial fibroid cancer womb",
    "alkylphenols": "nonylphenol industrial chemical endocrine disruptor estrogen surfactant",
    "berries": "blueberry antioxidant anthocyanin polyphenol fruit flavonoid",
    "mushrooms": "fungi ergothioneine antioxidant edible mushroom",
    "worms": "parasite helminth intestinal infection nematode",
    "toxins": "toxin poison chemical contaminant exposure heavy metal",
    "statin": "hmg coa reductase cholesterol lipid lowering ldl cardiovascular",
    "statins": "hmg coa reductase cholesterol lipid lowering ldl cardiovascular",
    "obesity": "overweight fat bmi adipose metabolic weight",
    "glaucoma": "eye intraocular pressure optic nerve vision blindness",
    "beans": "legume fiber protein bean plant",
    "seeds": "seed flax chia fiber omega nutrient",
    "soy": "soybean isoflavone phytoestrogen protein legume",
    "fiber": "dietary fiber fermentation colon bowel whole grain",
    "lard": "pork fat saturated animal cooking",
    "veal": "calf meat beef young",
}
