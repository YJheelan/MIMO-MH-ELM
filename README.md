# MIMO-MH-ELM
## Prévision Energétique Multi-Sources avec Réconciliation Optimale via une Approche Multi-Input Multi-Output Multi-Horizon (MIMO-MH) et Extreme Learning Machine (ELM)

La nature intermittente des énergies renouvelables pose plusieurs défis, notamment en matière de fiabilité, de qualité de l’énergie et d’équilibre entre l’offre et la demande [[1]](#ref1). Dans ce contexte, la prévision de la production d’électricité issue de sources renouvelables, telles que l’énergie éolienne et solaire, devient essentielle pour le fonctionnement efficace et continue du réseau électrique [[2]](#ref2). Dans ce contexte, l’approche Multi-Input Multi-Output Multi-Horizon (MIMO-MH) avec réconciliation et un Extrême Learning Machine (ELM) permet de synchroniser les prévisions des différentes sources (solaire, thermique, hydraulique, imports, etc.) pour garantir la cohérence avec la consommation nette (équivalente à la demande du réseau), tout en capturant les interactions et les contraintes physiques globales (équilibre offre-demande, import/export, autoconsommation) [[3]](#ref3). Contrairement aux modèles Single-Input Single-Output (SISO) qui traitent chaque source isolément, MIMO tire parti des corrélations entre sources et de la variabilité partagée, ce qui améliore la précision agrégée et réduit les écarts totaux (via l’effet de compensation entre erreurs). En parallèle, ELM apporte un apprentissage rapide, une solution fermée analytique et une faible charge computationnelle, le rendant idéal pour l’adaptation en quasi-temps réel. Cette approche optimise la prévision de la demande finale (consommation nette), indispensable pour le dispatch, la gestion des importations et la stabilité du réseau, tout en offrant une solution robuste et peu coûteuse pour des systèmes multi-énergies fortement variables et sujets à l’autoconsommation.



## Données Utilisées

Les performances des modèles de prévision énergétique dépendent directement de la qualité et de la représentativité des données. Pour capturer la dynamique des interactions entre les différentes sources d’énergie, nous utilisons des séries temporelles horaires détaillées, couvrant une période suffisante pour refléter la variabilité réelle. Une série temporelle est une suite d’observations indexée par le temps. Dans ce rapport, les séries temporelles représentent la production horaire d’électricité en MWh par différentes sources (Thermique, Hydraulique, Micro-hydraulique, Solaire photovoltaïque, Éolien, Bioénergies, Importations) 10. Elles incluent également le coût moyen de production en e/MWh et la production totale en MWh. Ces séries sont gérées par EDF sur la région Corse (https://opendata-corse.edf.fr/pages/home0/), couvrent la période 2016–2022 avec une résolution horaire, garantissant leur fiabilité et la représentativité du contexte énergétique insulaire corse.

## État d'Avancement (16/06/2025)
### Code Disponible
Les codes actuellement disponibles incluent :

- MIMO-ELM : architecture Multi-Input Multi-Output avec ELM
- Analyse de données : traitement et visualisation des données énergétiques

### Développements à Réaliser
Les fonctionnalités restant à implémenter sont :

- Time GPT (https://github.com/Nixtla/nixtla) : pour obtenir une référence et faire un benchmark
- Multi-Horizon (MH) : extension de MIMO-ELM pour la prévisions à différents horizons temporels
- Réconciliation : mécanismes d'ajustement pour garantir la cohérence des prévisions


## Références
**<a id="ref1">[1]</a>** Sheraz Aslam, Herodotos Herodotou, Syed Muhammad Mohsin, Nadeem Javaid, Nouman Ashraf, and Shahzad Aslam. [A survey on deep learning methods for power load and renewable energy forecasting in smart microgrids](https://doi.org/10.1016/j.rser.2021.110992). Renewable and Sustainable Energy Reviews, 144 :110992, 2021-07-
01.

**<a id="ref2">[2]</a>** Gilles Notton, Marie-Laure Nivet, Cyril Voyant, Christophe Paoli, Christophe Darras, Fabrice Motte, and Alexis Fouilloy. [Intermittent and stochastic character of renewable energy sources : Consequences, cost of intermittence and benefit of forecasting](https://doi.org/10.1016/j.rser.2018.02.007). Renewable and Sustainable Energy Reviews, 87 :96–105, 2018.

**<a id="ref3">[3]</a>** Cyril Voyant, Milan Despotovic, Gilles Notton, Yves-Marie Saint-Drenan, Mohammed Asloune, and Luis Garcia-Gutierrez. [On the importance of clearsky model in short-term solar radiation forecasting](https://doi.org/10.1016/j.solener.2025.113490). Solar Energy, 294 :113490, 2025.
