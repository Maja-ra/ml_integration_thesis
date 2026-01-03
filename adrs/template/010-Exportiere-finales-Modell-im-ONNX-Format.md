---
# source: https://github.com/adr/madr/blob/4.0.0/template/adr-template.md?plain=1
# options: https://adr.github.io/adr-templates/
---

# 010 Exportiere finales Modell im ONNX-Format

## Kontext and Problem-Statement

Das Vorhersagemodell wird gespeichert und anschließend in einer Runtime initialisiert, um Vorhersagen zu machen. In welchen Format wird das Modell abgelegt? Wie wird die Inferenz durchgeführt? 

<!-- This is an optional element. Feel free to remove. -->
## Entscheidungstreiber

* Plattformunabhängigkeit
* Consistency
* Sicherheit
* Inferenzzeit
* Dateigröße und Intitalisierungsdauer

## Betrachtete Optionen

* Exportiere finales Modell im ONNX-Format
* Exportiere finales Modell im Pickle-Format
* Exportiere finales Modell im -Format

## Entscheidungsergebnis

Ausgewählte Option: "Exportiere finales Modell im ONNX-Format", weil das Fromat zusätzlich zu guter Performance bei der Initialisierung und Inferenz unabhängig von der Plattform konsistent funktioniert sowie weniger Sicherheitslücken ausfweist. 

<!-- This is an optional element. Feel free to remove. -->
### Konsequenzen

* Gut, weil es die Sicherheit verbessert.
* Gut, weil es die Flexibilität verbessert.
* Gut, weil es die Zuverlässigkeit verbessert.
* Gut, weil es die Performance verbessert.
* Schlecht, weil zusätzliches Risiko für Fehler durch Transformationen entsteht.