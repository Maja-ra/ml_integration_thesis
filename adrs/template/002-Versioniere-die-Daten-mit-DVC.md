---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 002 Versioniere die Daten mit DVC

## Kontext and Problem-Statement

Input-Daten und Daten-Artefakte änderen sich oft in einem iterativen MLOps-Prozess, wodurch die Rückverfolgbarkeit und Reproduceability von Ergebnissen schwierig ist. Mit welchen Daten wurde Ein Modell trainiert? Kann eine ältere Version der Daten wiederhergestellt werden? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Speicherplatz
* Komplexität der Prozessintegration
* Wiederherstellung älterer Versionen

## Betrachtete Optionen

* Vollständige Daten und Artefakte auf Git hochladen
* Datenversionen mit DVC tracken

## Entscheidungsergebnis

Ausgewählte Option: "Datenversionen mit DVC tracken", weil Änderungen der Daten erkannt werden, DVC-Dateien nur die Änderung beinhalten und die Funktionen ähnlich wie Git in den Versionierungsprozess integriert werden können. 

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Adaptability verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.
* Gut, weil es die Reproduzierbarkeit verbessert.
