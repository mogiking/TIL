## my.cnf

### 한글 허용
기본 캐릭터 셋을 바꾸는 옵션.
```
# Character set (utf8mb4)
character-set-server            = utf8mb4
collation_server                = utf8mb4_general_ci
character_set-client-handshake  = FALSE
init_connect                    = set collation_connection=utf8mb4_general_ci
init_connect                    = set names utf8mb4
```
### Datadir 변경
```sql
> select @@datadir; #datadir 확인
```
```bash
> systemctl stop mariadb
> vi /etc/my.cnf #datadir 변경
> rsync -av [sourcedir] [targetdir]
> chown -R mysql:mysql [targetdir]
> systemctl start mariadb
```


## Query

### 캐릭터셋 확인
한글이 제대로 설정되고 있는지 확인
```sql
show variables like 'c%';
```

### 스키마 확인
```sql
show columns from [Tablename]
```

### 유저 추가
유저 확인
```sql
USE mysql;
SELECT host,user,password FROM user;
```

유저 생성 및 권한 부여
```sql
-- '%'는 외부 접속 허용
-- 'localhost'는 local 접속 허용
CREATE USER 'UserName'@'%' identified by 'Password';
-- 특정 Database의 모든 테이블 접근 허용
GRANT ALL PRIVILEGES ON [DatabaseName].* TO 'Username'@'%';
-- 
```