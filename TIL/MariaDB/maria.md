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
show columns from [Tablename];
show full columns from [Tablename];
```


## 설정

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


## Update MariaDB

CentOS7의 Default Repo를 사용하여 MariaDB를 설치하면 5.5 버전으로 설치가 된다.

Trigger를 Datetime에서 사용하지 못하는 등의 문제가 있기 때문에 최신 버전인 10.6 으로 업데이트를 진행했다.

Add MariaDB.repo at /etc/yum.repos.d/

```
[mariadb]
name=MariaDB
baseurl = https://mirror.yongbok.net/mariadb/yum/10.6/centos7-amd64
gpgkey=https://mirror.yongbok.net/mariadb/yum/RPM-GPG-KEY-MariaDB
gpgcheck=1
```

update mariadb

```bash
systemctl stop mariadb
yum update mariadb-server
systemctl start mariadb
```