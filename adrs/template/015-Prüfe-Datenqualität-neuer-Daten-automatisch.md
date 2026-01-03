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

## Entscheidungsergebnis

Ausgewählte Option: "Prüfe Datenqualität neuer Daten automatisch", weil dadurch eine zuverlässige Grundlage für Modelltraining gesichert wird und manuelle Arbeitsschritte entfallen. 

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Performance verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.