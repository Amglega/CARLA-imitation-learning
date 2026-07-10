#!/bin/bash

# IP local y subred
IP_LOCAL=$(hostname -I | awk '{print $1}')

SUBRED_ACTUAL=$(echo $IP_LOCAL | cut -d. -f1-3).0/24

echo "--- Configuración de Firewall del DeepRacer---"
echo "IP detectada: $IP_LOCAL"
echo "Subred actual: $SUBRED_ACTUAL"

# Limpiar reglas antiguas
echo "Buscando reglas antiguas"

REGLAS_ANTIGUAS=$(sudo ufw status | grep '/24' | awk '{print $3}' | sort -u)

for regla in $REGLAS_ANTIGUAS; do
    if [ "$regla" != "$SUBRED_ACTUAL" ]; then
        echo "Eliminando regla: $regla"
        sudo ufw delete allow from "$regla" > /dev/null
    else
        echo "La regla para $regla ya existe y es la actual"
    fi
done

# Añadir la red actual
sudo ufw allow from "$SUBRED_ACTUAL" > /dev/null
echo "Regla aplicada con éxito para: $SUBRED_ACTUAL"

# Imprimir cómo ha quedado el firewall
echo "---------------------------------"
sudo ufw status
