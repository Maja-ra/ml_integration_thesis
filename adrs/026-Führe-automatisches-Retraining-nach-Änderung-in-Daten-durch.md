---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 026 Führe automatisches Retraining nach Änderung in Daten durch

## Kontext and Problem-Statement

Daten verändern sich, wodurch ein Drift entstehen kann. Ist ein Retraining notwendig? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Sicherstellung Modell-Performance

## Betrachtete Optionen

* Führe automatisches Retraining nach Änderung in Daten durch
* Führe periodisch Retraining durch

## Entscheidungsergebnis

Ausgewählte Option: "Führe automatisches Retraining nach Änderung in Daten durch", weil dadurch das Modell immer an die aktuellen Gegebenheiten angepasst ist.  

## Beinflussende Empfehlungen

* 8.16
* 9.9

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Performance verbessert.
* Gut, weil es die Datenqualität verbessert.
* Gut, weil es die Modellgüte verbessert.
* Gut, weil es die Anpassungsfähigkeit verbessert.
* Schlecht, weil es die Komplexität erhöht.
* Schlecht, weil es das Risiko erhöht.