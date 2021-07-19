# Latex
선형대수학 수업을 들으며 기말고사 과제의 제출을 위해 반 강제적으로 LaTex를 시작하게 되었다. 무엇이든 새로운 것을 접할땐 초기에는 재밌지만 역시 익숙해져가는 과정은 쉽지 않았다. 수식들이 이쁘게 적히는 것이 신기했지만, 역으로 이쁘게 만들기 위해서 웹을 뒤져야하는 것이 다소 불편했다. 
LaTex를 조금 더 이쁘게 만들기 위해, 작은 wiki의 형태로 latex_wiki.md를 작성하려고 한다.

과제 제출 당시에는 [Overleaf](https://www.overleaf.com/)라는 온라인 에디터를 사용했지만, 최근에는 notion의 Inline Equation, Block Equation을 더 자주 사용하게 된다.

때문에 최근 더 자주 사용하고 있는 notion의 Equation을 기준으로 새로운 기호를 사용할 때마다 작성을 하도록 한다.

## Equation in Github
Github Markdown에선 Link의 이미지를 가져오는 방법으로 구현한다. 본 문서에선 [iTex2Img](http://www.sciweavers.org/free-online-latex-equation-editor)의 서비스를 이용하여 구현하도록 한다. 
예시는 아래의 링크 중 **[YOUR EQUATION]** 부분에 TeX형식으로 입력을 하면 된다.
```
![equation](http://www.sciweavers.org/tex2img.php?eq=[YOUR EQUATION]&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)
```
**Example**: 
```
Q^*(s,a)=R^{a}_{s}+\gamma\underset{s'\in{S}}{\sum}P^{a}_{ss'},(\underset{a}{max}Q^{*}({s'},{a'}))
```
**Outcome**:
![equation](http://www.sciweavers.org/tex2img.php?eq=Q^*(s,a)=R^{a}_{s}+\gamma\underset{{s'}\in{S}}{\sum}P^{a}_{ss'},(\underset{a}{max}Q^{*}({s'},{a'}))&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=)

