---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 028 Nutze MySQL Datenbank

## Kontext and Problem-Statement

Generierte Daten werden konsolidiert. In welcher Datenbank? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Daten-Format
* Aufbau Open-Spource-Software

## Betrachtete Optionen

* Nutze MySQL Datenbank
* Nutze PostreSQL Datenbank
* Nutze MongoDb Datenbank

## Entscheidungsergebnis

Ausgewählte Option: "Nutze MySQL Datenbank", weil dadurch das Modell immer an die aktuellen Gegebenheiten angepasst ist.  

## Beinflussende Empfehlungen

* 13.1

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Nachvollziehbarkeit verbessert.
* Gut, weil es die Datenintegrität verbessert.
* Schlecht, weil es die Performance verschlechtert.