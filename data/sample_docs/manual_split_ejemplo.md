# Manual de Servicio Técnico - Climatizador Split Inverter Serie EcoPro 12K

## 1. Especificaciones Eléctricas y de Refrigeración
- **Modelo:** ECO-12INV-R32
- **Capacidad:** 12,000 BTU/h (1 Ton)
- **Voltaje de Alimentación:** 220V - 240V AC, 1 Fase, 60 Hz
- **Refrigerante:** R-32 (Carga de fábrica: 650 gramos)
- **Presión de Succión Normal (Modo Frío):** 115 - 130 PSI a 35°C exterior
- **Consumo Nominal:** 4.8 Amperios

## 2. Códigos de Falla y Autodiagnóstico

### Código E6: Bloqueo o Falla del Motor Ventilador Exterior
- **Descripción Oficial:** Anomalía en la retroalimentación de velocidad (señal Hall) del motor ventilador de la unidad exterior o rotor bloqueado.
- **Causas Probables:**
  1. Conector del motor desconectado o terminal sulfatado en la placa PCB.
  2. Aspa del ventilador trabada por cuerpo extraño o cojinetes atascados.
  3. Devanado del motor abierto o en cortocircuito.
  4. Falla en el circuito de control de la PCB exterior.
- **Procedimiento de Medición y Diagnóstico:**
  1. Desenergizar la unidad completamente y esperar 3 minutos a que los condensadores electrolíticos se descarguen.
  2. Girar el aspa manualmente: debe girar con suavidad sin rozamientos mecánicos.
  3. Desconectar el arnés del motor (conector de 5 pines) de la PCB.
  4. Medir resistencia entre los devanados U-V, V-W, W-U con multímetro en escala de ohmios (debe marcar entre 45Ω y 55Ω balanceados en los tres pares). Si algún devanado marca circuito abierto (OL) o resistencia desbalanceada mayor al 10%, reemplazar el motor ventilador.
  5. Si las resistencias están correctas y el voltaje de alimentación al conector es de 310V DC entre pin 1 (Rojo) y pin 3 (Negro), pero no hay rotación, la falla se localiza en la placa de control PCB.

### Código F1: Falla del Sensor de Temperatura de Tubería (Termistor)
- **Descripción Oficial:** Sensor de temperatura del serpentín evaporador descalibrado, abierto o en cortocircuito.
- **Valor Nominal de Referencia:** Termistor NTC tipo 10kΩ a 25°C.
- **Valores de Resistencia según Temperatura:**
  - A 20°C: 12.5 kΩ
  - A 25°C: 10.0 kΩ
  - A 30°C: 8.1 kΩ
- **Acción Correctiva:** Medir la resistencia con termómetro acoplado. Si la lectura discrepa en más de 1.5 kΩ del valor nominal, sustituir el termistor.

## 3. Procedimiento de Vacío y Prueba de Hermeticidad
1. Conectar bomba de vacío de doble etapa (mínimo 5 CFM) y vacuómetro digital calibrado a la toma de servicio de baja presión.
2. Evacuar el sistema hasta alcanzar un nivel inferior a 500 micrones de mercurio (Hg).
3. Cerrar la válvula del manómetro y apagar la bomba de vacío.
4. Realizar prueba de retención durante un mínimo de 15 minutos: la presión no debe elevarse por encima de 800 micrones. Un incremento rápido indica fuga activa en abocardados.
