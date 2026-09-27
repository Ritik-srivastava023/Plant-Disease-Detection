"""
Comprehensive Agronomic Disease Knowledge Base for PlantVillage (15 Classes).
Provides symptom descriptions, pathogen classifications, and actionable treatment recommendations.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "pathogen": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "symptoms": (
            "Small, water-soaked, dark-brown or black angular spots on leaves. "
            "Lesions may develop chlorotic yellow halos. Severe infection causes extensive defoliation, "
            "exposing fruits to sunscald."
        ),
        "treatment": [
            "Apply fixed copper fungicides combined with mancozeb for synergistic suppression.",
            "Use certified pathogen-free seeds and hot-water treated transplant plugs.",
            "Incorporate bio-control agents like Bacillus subtilis or Streptomyces-based bio-bactericides."
        ],
        "prevention": [
            "Avoid overhead irrigation to minimize bacterial spread across wet leaf canopies.",
            "Enforce a 2-3 year crop rotation away from solanaceous species (tomato, pepper, eggplant).",
            "Sanitize all pruning tools, stakes, and field machinery regularly."
        ]
    },
    "Pepper__bell___healthy": {
        "crop": "Bell Pepper",
        "condition": "Healthy Leaf Tissue",
        "status": "Healthy",
        "pathogen": "None (Optimum Physiological Health)",
        "symptoms": (
            "Vibrant green leaf lamina with uniform chlorophyll distribution, intact cuticle, "
            "and absence of necrotic or chlorotic lesions."
        ),
        "treatment": [
            "No chemical or corrective treatment needed.",
            "Maintain balanced N-P-K fertigation suited to the vegetative/fruiting stage."
        ],
        "prevention": [
            "Continue periodic scouting for early aphid or thrip vectors.",
            "Maintain adequate soil moisture and calcium levels to prevent blossom end rot."
        ]
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "condition": "Early Blight",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": (
            "Characteristic concentric circular rings producing a 'target-board' pattern on older foliage. "
            "Spots turn dark brown to black and are surrounded by a prominent yellow chlorotic zone."
        ),
        "treatment": [
            "Apply protective fungicides (Chlorothalonil, Mancozeb) at early onset.",
            "Rotate systemic fungicides such as Azoxystrobin, Difenoconazole, or Boscalid to mitigate resistance.",
            "Apply copper oxychloride during humid weather conditions."
        ],
        "prevention": [
            "Maintain optimal nitrogen levels (stressed, nutrient-deficient plants are highly susceptible).",
            "Remove and incinerate infected crop debris following harvest.",
            "Employ drip irrigation to keep canopy moisture low."
        ]
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "condition": "Late Blight",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": (
            "Aggressive water-soaked pale-green to dark brown lesions expanding rapidly across foliage. "
            "A delicate white cottony fungal sporulation appears on the abaxial (underside) leaf surface during high humidity."
        ),
        "treatment": [
            "Immediately apply targeted anti-oomycete systemic fungicides (Metalaxyl / Mefenoxam, Dimethomorph, Cymoxanil).",
            "Apply protective multisite fungicides (Fluazinam, Chlorothalonil) ahead of forecasted cool, wet weather.",
            "Prune and destroy severely infected vines to halt spore dispersal."
        ],
        "prevention": [
            "Plant certified disease-free seed tubers with documented resistance.",
            "Eliminate cull piles and volunteer potato sprouts within the vicinity.",
            "Monitor regional late blight forecasting models (e.g., BLITECAST)."
        ]
    },
    "Potato___healthy": {
        "crop": "Potato",
        "condition": "Healthy Leaf Tissue",
        "status": "Healthy",
        "pathogen": "None (Optimum Physiological Health)",
        "symptoms": (
            "Well-developed compound leaf structure, uniform green coloration, and active photosynthesis without foliar distress."
        ),
        "treatment": [
            "No chemical intervention required.",
            "Sustain regular soil hilling to shield developing tubers from sunlight."
        ],
        "prevention": [
            "Maintain preventive foliar spray schedule during wet, overcast weather.",
            "Test soil potassium and phosphorus availability periodically."
        ]
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato",
        "condition": "Bacterial Spot",
        "status": "Diseased",
        "pathogen": "Bacterium (Xanthomonas perforans / euvesicatoria)",
        "symptoms": (
            "Numerous small (1-3 mm), greasy, dark brown necrotic spots on leaflets. "
            "Lesions coalesce, causing leaf scorch, blighting, and extensive defoliation."
        ),
        "treatment": [
            "Foliar sprays containing copper bactericides mixed with ethylene bisdithiocarbamate (EBDC / Mancozeb).",
            "Introduce systemic acquired resistance (SAR) activators such as Acibenzolar-S-methyl.",
            "Apply bacteriophages targeting Xanthomonas strains where commercially available."
        ],
        "prevention": [
            "Use certified pathogen-tested seeds; treat seed lots with 1.3% sodium hypochlorite.",
            "Avoid field operations and pruning when plants are wet with dew or rain.",
            "Disinfect greenhouse stakes, wire supports, and seedling trays."
        ]
    },
    "Tomato_Early_blight": {
        "crop": "Tomato",
        "condition": "Early Blight",
        "status": "Diseased",
        "pathogen": "Fungus (Alternaria solani / A. tomatophila)",
        "symptoms": (
            "Dark brown concentric necrotic rings forming a 'bullseye' lesion on lower mature leaves. "
            "Progresses upwards, causing premature senescence, leaf dropping, and collar rot on stems."
        ),
        "treatment": [
            "Foliar applications of Strobilurins (Pyraclostrobin, Azoxystrobin) or SDHI fungicides (Boscalid).",
            "Apply preventative broad-spectrum protectants such as Chlorothalonil or Mancozeb every 7-10 days.",
            "Spray copper-based organic formulations (Bordeaux mixture) in eco-certified farms."
        ],
        "prevention": [
            "Stake and cage indeterminate vines to elevate foliage above soil splash.",
            "Mulch beds with straw or plastic film to block soilborne spore dispersal.",
            "Prune lowest 30 cm of foliage as plants mature to enhance airflow."
        ]
    },
    "Tomato_Late_blight": {
        "crop": "Tomato",
        "condition": "Late Blight",
        "status": "Diseased",
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": (
            "Large, irregularly shaped, dark olive-brown to greasy black lesions on foliage and petioles. "
            "White velvety sporulation develops on the leaf undersides in humid conditions; leads to total plant collapse within days."
        ),
        "treatment": [
            "Apply translaminar / systemic fungicides (Mandipropamid, Propamocarb, Fluopicolide, Dimethomorph).",
            "Immediately isolate and destroy infected vines to safeguard neighboring crops.",
            "Apply protective copper fungicides ahead of rain episodes."
        ],
        "prevention": [
            "Select resistant cultivars (e.g., Mountain Magic, Defiant PHR, Plum Regal).",
            "Ensure full sun exposure and wide plant spacing (at least 60 cm) for fast canopy drying.",
            "Destroy any nearby volunteer tomato and potato plants."
        ]
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato",
        "condition": "Leaf Mold",
        "status": "Diseased",
        "pathogen": "Fungus (Passalora fulva / Fulvia fulva)",
        "symptoms": (
            "Pale green to distinct bright yellow chlorotic patches on the upper leaf surface. "
            "Concurrently, olive-green to velvet brown velvety fungal sporulation coats the corresponding underside."
        ),
        "treatment": [
            "Fungicidal applications of Difenoconazole, Cyazofamid, or Boscalid.",
            "Apply potassium bicarbonate or sulfur-based sprays for organic suppression."
        ],
        "prevention": [
            "Primary greenhouse issue: reduce relative humidity below 85% through aggressive ventilation and heating.",
            "Utilize drip irrigation exclusively and prune dense inner canopy foliage.",
            "Choose resistant hybrids possessing Cf gene resistance."
        ]
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato",
        "condition": "Septoria Leaf Spot",
        "status": "Diseased",
        "pathogen": "Fungus (Septoria lycopersici)",
        "symptoms": (
            "Prolific tiny circular spots (2-4 mm) with dark brown margins and sunken grayish-white centers. "
            "Tiny black pycnidia (fruiting bodies) resemble pepper grains within the lesion core."
        ),
        "treatment": [
            "Routine applications of Chlorothalonil, Mancozeb, or Copper hydroxide starting at first bloom.",
            "Incorporate QoI fungicides (Azoxystrobin) or DMI fungicides (Difenoconazole) upon symptom emergence."
        ],
        "prevention": [
            "Deep plow or compost infected crop residues after harvest to bury overwintering pycnidia.",
            "Maintain strict 3-year crop rotations with non-solanaceous crops.",
            "Apply organic mulch around plant bases to prevent fungal splash from rain."
        ]
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato",
        "condition": "Two-Spotted Spider Mite Infestation",
        "status": "Diseased",
        "pathogen": "Arachnid Pest (Tetranychus urticae)",
        "symptoms": (
            "Fine pale-yellow or white stippling / speckling across the upper leaf surface. "
            "Silken fine webbing develops beneath leaves and across apical shoots during heavy infestations, causing leaf bronze and desiccation."
        ),
        "treatment": [
            "Apply selective acaricides / miticides (Abamectin, Bifenazate, Spiromesifen, Hexythiazox).",
            "Introduce predatory beneficial mites (Phytoseiulus persimilis or Neoseiulus californicus).",
            "Spray insecticidal soaps or cold-pressed neem oil ensuring thorough lower-canopy coverage."
        ],
        "prevention": [
            "Maintain field humidity and avoid excessive dry, dusty roadside conditions.",
            "Refrain from over-applying broad-spectrum synthetic pyrethroids which eradicate natural mite predators.",
            "Regularly rinse leaf undersides with pressurized water in smallholdings."
        ]
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato",
        "condition": "Target Spot",
        "status": "Diseased",
        "pathogen": "Fungus (Corynespora cassiicola)",
        "symptoms": (
            "Small, pinpoint lesions that expand into circular brown necrotic spots with subtle concentric zones. "
            "Unlike Early Blight, spots remain smaller and are accompanied by extensive upper canopy yellowing."
        ),
        "treatment": [
            "Fungicides containing Fluxapyroxad, Boscalid, Famoxadone + Cymoxanil, or Chlorothalonil.",
            "Apply Azoxystrobin alternating with multi-site contact protectants."
        ],
        "prevention": [
            "Ensure wide spacing to facilitate rapid drying of foliage after dew.",
            "Eliminate alternative host weeds (such as nightshade and morning glory).",
            "Avoid planting adjacent to mature cotton, soybean, or cucumber fields which harbor Corynespora."
        ]
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato",
        "condition": "Yellow Leaf Curl Virus (TYLCV)",
        "status": "Diseased",
        "pathogen": "Begomovirus (Vectored by Bemisia tabaci / Sweetpotato Whitefly)",
        "symptoms": (
            "Severe upward leaf curling, cupping, and puckering. Marked interveinal chlorosis (yellowing) "
            "and severe plant stunting. Flowers abort, leading to drastic fruit yield loss."
        ),
        "treatment": [
            "No curative chemical therapy exists for viral infections; eradicate infected plants immediately.",
            "Control the whitefly vector using systemic insecticides (Imidacloprid, Thiamethoxam, Cyantraniliprole, Spirotetramat).",
            "Apply insecticidal soap or horticultural mineral oils to smother whitefly nymphs."
        ],
        "prevention": [
            "Plant TYLCV-resistant or tolerant cultivars carrying Ty-1, Ty-2, or Ty-3 resistance genes.",
            "Install 50-mesh insect-proof screening on all greenhouse vents.",
            "Use yellow sticky cards for whitefly population monitoring.",
            "Maintain a 2-month host-free crop break between tomato cropping cycles."
        ]
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato",
        "condition": "Tomato Mosaic Virus (ToMV)",
        "status": "Diseased",
        "pathogen": "Tobamovirus (Mechanically / Seed Transmitted)",
        "symptoms": (
            "Mottled light and dark green mosaic patterns across foliage. Distortion such as 'fern-leaf' or 'shoestring' leaf tapering. "
            "Internal browning of fruit vascular walls in severe infections."
        ),
        "treatment": [
            "No chemical cure exists for viral plant diseases; rogue out and burn all symptomatic plants.",
            "Dip tools and hands in nonfat dry milk (20%) or 10% trisodium phosphate (TSP) when handling plants."
        ],
        "prevention": [
            "Plant certified virus-free seed treated with heat (70°C for 3 days) or trisodium phosphate.",
            "Prohibit tobacco use (smoking/chewing) anywhere near tomato plants to avoid mechanical transmission.",
            "Sterilize greenhouse benches, pruning shears, and stakes between planting seasons."
        ]
    },
    "Tomato_healthy": {
        "crop": "Tomato",
        "condition": "Healthy Leaf Tissue",
        "status": "Healthy",
        "pathogen": "None (Optimum Physiological Health)",
        "symptoms": (
            "Lush green pinnate leaflets exhibiting vigorous turgidity, uniform coloration, "
            "unobstructed stomatal conductance, and zero pathological lesions."
        ),
        "treatment": [
            "No corrective treatment required.",
            "Continue optimal micronutrient feeding (Zinc, Boron, Magnesium) to sustain vigor."
        ],
        "prevention": [
            "Maintain prophylactic bio-fungicide sprays (Trichoderma harzianum soil drench).",
            "Conduct weekly visual scouting across the field."
        ]
    }
}

