---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 018 Speichere Vorhersageergebnisse mit Abhängigkeiten

## Kontext and Problem-Statement

Das Modell erstellt eine Vorhersage basierend auf den Input-Daten. Was wurde vorhergesagt? Von welchem Modell, basierend auf welchen Daten?  

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Transparenz Modellverhalten
* Auswertung entstandener Daten

## Betrachtete Optionen

* Speichere Vorhersageergebnisse mit Abhängigkeiten
* Logge Vorhersageergebnisse ausschließlich
* Speichere Vorhersageergebnisse nicht

## Entscheidungsergebnis

Ausgewählte Option: "Speichere Vorhersageergebnisse mit Abhängigkeiten", weil das Modellverhalten dadurch überwacht und neue Auswertungen erstellt werden können.  

## Beinflussende Empfehlungen

* 9.7
* 11.4

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Wiederverwendbarkeit verbessert.
* Gut, weil es die Nachvollziehbarkeit verbessert.