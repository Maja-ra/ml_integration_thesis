---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 015 Prüfe Datenqualität neuer Daten automatisch

## Kontext and Problem-Statement

Viel Fehelerpotenzial existiert in Datensätzen, von denen das die Modellperformance abhängig ist. Wird die Datenqualität überprüft? Wie und wann?

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Starke Datenabhängigkeit
* Viele Datenpunkte
* Performance und Ressourceneffizienz

## Betrachtete Optionen

* Prüfe Datenqualität neuer Daten automatisch
* Gewährleiste Datenqualität vor dem Einspeisen
* Überprüfe Datenqualität nicht

## Beinflussende Empfehlungen

Ausgewählte Option: "Prüfe Datenqualität neuer Daten automatisch", weil dadurch eine zuverlässige Grundlage für Modelltraining gesichert wird und manuelle Arbeitsschritte entfallen. 

## Betroffene Empfehlungen

* 3.1
* 3.2
* 3.12
* 4.12
* 4.18
* 8.4
* 13.5
* 13.8

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Performance verbessert.
* Gut, weil es die Datenqualität verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
