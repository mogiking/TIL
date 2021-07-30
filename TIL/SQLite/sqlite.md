# SQLite
추가로 시스템을 설치하기는 힘들고, 리소스도 여유있지 않은 프로젝트에서 SQLite를 임시로 사용하여 임시데이터를 저장하기로 했다.
기존에 알고 있던 DBMS와 Query 등은 비슷하지만 단일 파일 형태로 DB를 구성한다는 것이 흥미로웠다.
저장용량: 140TB. 오히려 저장소의 리소스가 부족할 것 같다.
입출력과 데이터관리에 대해서 확인해야겠지만 유용하게 사용할 수 있을 것 같다.

## Table list 확인
```SQL
SELECT name FROM sqlite_master WHERE type = 'table'
```