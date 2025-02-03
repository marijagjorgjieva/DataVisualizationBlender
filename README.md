Овој додаток е создаден како дел од предметот Компјутерска анимација на ФИНКИ и овозможува визуелизација на различни типови на податоци директно во 3D просторот на Blender. 
# Документација
&emsp;Деталната документација за додатокот можете да ја најдете <a href="https://github.com/marijagjorgjieva/DataVisualizationBlender/wiki">овде</a> (ﾉ◕ヮ◕)ﾉ*:･ﾟ✧
<div style="display: flex; justify-content: space-between; align-items: center;">
        <img src="https://github.com/user-attachments/assets/e4e90dc1-f672-4736-b8a8-7395c8dee8a2" alt="Image 2" style="width: 30%; height: auto;">
        <img src="https://github.com/user-attachments/assets/351448e5-6c5b-47e5-99d8-8c6fe37d5639" alt="Image 3" style="width: 30%; height: auto;">
                <img src="https://github.com/user-attachments/assets/4d79fd76-05fe-4212-8130-cb4641334f28" alt="Image 1" style="width: 30%; height: auto;">

</div>

# Функционалности

Овој додаток нуди различни алатки за визуелизација, вклучувајќи:
## Импортирање и визуелизација на податоци CSV датотеки со формат на колони X,Y,(Z)

* Креирање хистограми – Генерирање на хистограм со прилагодлив број на колони и бои.

* Бар графици – Претставување на податоци во форма на столбести графици со можност за конфигурирање на ширина и боја.

* Дијаграми за расејување (Scatter Plot) – Визуелизација на податоци во вид на точки во 3D простор.

* Линиски графици (Line Plot) – Креирање на графикон со линии за претставување на трендови во податоците.

* Оски (Axis Creator) – Автоматско додавање на X, Y и Z оски со прилагодливи бои.

## Импортирање на текстуални датотеки – Анализа на текстови и креирање на:

* Word Frequency Clouds – Визуелизација на најчестите зборови во текст.
<div style="width: 100%; height: 100vh; display: flex; justify-content: center; align-items: center;">
    <img src="https://github.com/user-attachments/assets/85995e61-375a-4b11-bf9e-362ea227567b" alt="Image" style="width: 300px;">
</div>
* TF-IDF Word Clouds – Прикажување на најрелевантните зборови базирано на TF-IDF алгоритам.

* Topic Modeling Clouds – Генерирање облаци на зборови базирани на теми извлечени со Latent Dirichlet Allocation (LDA).

# Инсталација

Превземете го додатокот од официјалниот GitHub репозиториум.

Отворете го Blender и одете во Edit > Preferences > Add-ons.

Кликнете на Install и изберете го .zip фајлот на додатокот.

Активирајте го додатокот со означување на чекбоксот.
За користење на дел од алатките за визуелизација на текстуалните фајлови потребно е да ја инсталирате библиотеката scikit-learn во околината на Blender.

# Упатство за користење

1. Додавање на графикон

Откако додатокот е активиран, во 3D Viewport, притиснете N за да го отворите Sidebar.

Изберете ја Data Visualization табот.

Изберете го типот на графикон и внесете ги потребните податоци.

2. Уредување на графикон

Користете го панелот за подесувања за промена на бои, големини и распоред на елементите.

3. Експортирање

Можете да го зачувате финалниот графикон како .blend фајл или да го рендерирате како слика/анимација.
<br><br>

<img src="https://user-images.githubusercontent.com/82838042/177052296-0b90e46b-1859-4222-9386-d6936a976aff.gif" alt="Customize" align="right" width="150"/> <br /><br/>

