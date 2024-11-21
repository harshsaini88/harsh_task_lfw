import os

# Configuration file for paths and settings
class Config:
    DATASET_PATH = os.path.join("data", "lfw")
    IMG_SIZE = 64  # Resize images to 64x64
    CATEGORIES = ["Category_A", "Category_B"]
    
    # 10 example names for Category A
    CATEGORY_A_NAMES = [
        "George_W_Bush", "Tony_Blair", "Donald_Rumsfeld", "Ariel_Sharon", 
        "Gerhard_Schroeder", "Jacques_Chirac", "Junichiro_Koizumi", 
        "Luiz_Inacio_Lula_da_Silva", "Hugo_Chavez", "Jean_Chretien"
    ]
    
    # 10 example names for Category B
    CATEGORY_B_NAMES = [
        "Colin_Powell", "Condoleezza_Rice", "Vladimir_Putin", "John_Ashcroft",
        "Pervez_Musharraf", "Alvaro_Uribe", "Abdullah_Gul", "Silvio_Berlusconi",
        "Paul_Bremer", "Kofi_Annan"
    ]
    
    BATCH_SIZE = 32
    EPOCHS = 10
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.1
    RANDOM_STATE = 42
