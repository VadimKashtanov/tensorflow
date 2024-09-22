rm meilleur_model.h5.keras dernier__model.h5.keras
set -e
clear
#python3 cree_les_données.py
python3 cnn.py
python3 tester_la_validitée.py meilleur_model.h5.keras
#python3 tester_la_validitée.py dernier__model.h5.keras