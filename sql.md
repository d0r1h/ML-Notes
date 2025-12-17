---
icon: server
---

# SQL

[DataLemur](https://datalemur.com/) | [Interview Query](https://www.interviewquery.com/) | [InterviewBit](https://www.interviewbit.com/) | [StrataScratch](https://www.stratascratch.com/) | [Deep-ML](https://www.deep-ml.com/)

DBMS | DDL | DML | Key Attributes | Constraints | SET Operators | Functions | Joins | Sub-Queries | Data Integrity | Clause

**DBMS :**: DBMS stands for database management system, there are systems to store, retrieve, and manipulate data it can handle large amounts of data.&#x20;

There are 4 different cardinalities (relationship):

1. One-to-one relationship&#x20;
2. Many-to-one relationship
3. One-to-many relationship
4. Many-to-many relationship

**RDMBS** : RDMBS stands for the relational database management system. RDBMS allows for storage, retrieving, and manipulation of data, but in a more efficient way than DBMS. Apart from rows and columns, the RDBMS table has the following components:

* Domain
* Instance
* Schema
* Keys



A database consists of one or more tables, a table is the most significant component in an RDMBS, and a table consists of rows and columns. Each column represents the attributes of the entity.&#x20;

* Each row in a table is a record/tuple.&#x20;
* Each column in a table is an attribute

A key is a data item (a column or a set of columns) to uniquely identify a record in a table.&#x20;

1. Candidate key
2. Primary Key
3. Foreign Key
4. Unique Key
5. Alternate Key&#x20;

A domain is a set of values that an attribute can take and it won’t accept values outside of its domain.&#x20;

A database schema is a blueprint that represents the logical view of the database. It defines the tables and the relationship between them.

1. Physical Schema: -  How data is stored in actual storage described at this level
2. Logical Schema: - Pertains the logical design that needs to be applied to the data stored.&#x20;

**A candidate Key** can be any column or a combination of columns that can qualify as a unique key in the database. There can be multiple Candidate Keys in one table. Each Candidate Key can qualify as a Primary Key.\
**A Primary Key** is a column or a combination of columns that uniquely identifies a record. There is only one Primary Key in a table

**DataBase**

* A database consists of one or more tables
* Each row in table called record or tuple
* Each column in a table is an attribute
* Keys - Data item to uniquely identify a record in a table ( Primary, Unique, Foreign … )

DataBase schema is a blueprint that represents the logical view of the database., defines tables and relationships between them.

<figure><img src=".gitbook/assets/unknown (1).jpeg" alt=""><figcaption></figcaption></figure>

Structured Query Language(SQL) as we all know is the database language by the use of which we can perform certain operations on the existing database and also we can use this language to create a database.

SQL commands are mainly categorized into five categories as:

1. **DDL** (Data Definition Language)  → Create, Alter, Drop, Truncate
2. **DML** (Data Manipulation Language) → Insert, Update, Delete
3. **DQL** (Data Query Language) → Select&#x20;
4. **DCL** (Data Control Language) → Grant, Revoke
5. **TCL** (Transactional Control Language) → Commit, Roll Back, Save Point, Set Transaction

<figure><img src=".gitbook/assets/unknown (3).png" alt=""><figcaption></figcaption></figure>

**Query order of execution** &#x20;

From  and Joins -> Where -> Group By -> Having -> Select -> Order By -> Limit / Offset

**Constraint**

Used for specifying the rules for data in the table, and can be specified when the table is created.

* Not Null
* Unique
* Primary Key
* Check
* Default

**Wildcard** **Filtering** --> %, -, \_, \[], ^, #

**Aggregate Functions :-** It’s all about performing calculations on multiple rows of a single column of a table and returning a single value. The Group By statement groups rows that have the same values into summary rows, and can contain two or more columns. \[Count, Sum, Avg, Max, Min]

**Where vs Having**&#x20;

1. Where filters the data at row level(an individual record) but Having clause filter the data at the group level(aggregation)
2. Having clause can’t be used without group by(because having works with aggregation and without groupby you cant get aggregate values) but where clause can be used without group by also.&#x20;
3. Where is used before the groupby() and Having is used after the groupby() clause.

**Windows Function**

Window functions use values from multiple rows to produce values for each row separately. What distinguishes window from other SQL functions, namely aggregate and scalar functions, is the keyword OVER, used to define a portion of rows the function should consider as inputs to produce an output. This portion of rows is called a ‘window’, hence the name of this family of functions. Within the OVER() we use clauses.

The main difference between the standard aggregate function and the window function is that while the former reduces the number of rows to match the number of categories to which the data are aggregated, the latter does not change the number of rows, instead assigns the correct value to each row of the dataset, even if these values are the same.

At a high-level window functions can be split into following 3 categories:

<figure><img src=".gitbook/assets/unknown (2) (1).png" alt=""><figcaption></figcaption></figure>

Work as group by but no of rows is not reduced (partition by), mostly used for ranking.&#x20;

**Ranking Window Functions:**

The ranking window functions are used to assign numbers to rows depending on some defined ordering.

1. **Row Number** :- The ROW\_NUMBER() function is the simplest of the ranking window functions in SQL. It assigns consecutive numbers starting from 1 to all rows in the table. The order of the rows needs to be defined using an ORDER BY clause inside the OVER clause. This is, in fact, the necessary condition for all the ranking window functions: they don’t require the PARTITION BY clause but the ORDER BY clause must always be there. The function simply assigns the consecutive numbers to the rows, without repeating or skipping any number.
2. **RANK** :- The RANK() window function is more advanced than ROW\_NUMBER(). The key difference between RANK() and ROW\_NUMBER() is how the first one handles the ties, i.e. cases when multiple rows have the same value by which the dataset is sorted. The rank function will always assign the same value to rows with the same values and will skip some values to keep the ranking consistent with the number of preceding rows. The RANK() function is especially useful when the task is to output the rows with the highest or the lowest values.
3. **Dense** **Rank** :- The DENSE\_RANK() function is very similar to the RANK() function with one key difference - if there are ties in the data and two rows are assigned the same ranking value, the DENSE\_RANK() will not skip any numbers and will assign the consecutive value to the next row. The DENSE\_RANK() is most useful when solving tasks in which several highest values need to be shown.
4. **PERCENT** **RANK** :- The PERCENT\_RANK() function is another one from the ranking functions family that in reality uses a RANK() function to calculate the final ranking. The ranking values in the case of PERCENT\_RANK are calculated using the following formula: (rank - 1)/(rows - 1). Because of this, all the ranking values are scaled by the number of rows and stay between 0 and 1. Additionally, the rows with the first value are always assigned the ranking value of 0.
5. **Ntile** :- It works analogically to the ROW\_NUMBER function but instead of assigning consecutive numbers to the next rows, it assigns consecutive numbers to the buckets of rows. The bucket is a collection of several consecutive rows and the number of buckets is set as a parameter of the NTILE() function - for example, NTILE(10) means that the dataset will be divided into 10 buckets.

**Value Window Functions**

1. **Lag** :- The LAG() function is by far the most popular out of the value window functions but at the same time is rather simple. What it does is, it assigns to each row a value that normally belongs to the previous row. In other words, it allows to shift any column by one row down and allows to perform queries using this shift of values. Naturally, the ordering of the rows matters also in this case, hence, the window function will most commonly include the ORDER BY clause within its OVER() clause.
2. **Lead** :- The LEAD() window function is the exact opposite of the LAG() function because while LAG() returns for each row the value of the previous row, the LEAD() function will return the value of the following row. In other words, as LAG() shifts the values 1 row down, LEAD() shifts them 1 row up. Otherwise, the functions are identical in how they are called or how the order is defined.
3. **First** **Value** :- The FIRST\_VALUE() function is not that commonly used but is also a rather interesting value window function in SQL. It does exactly what its name suggests - for all the rows, it assigns the first value of the table or the partition to which it is applied, according to some ordering that determines which row comes as a first one. Moreover, the variable from which the first value should be returned needs to be defined as the parameter of the function.
4. **Last** **Value** :- The LAST\_VALUE() function is the exact opposite of the FIRST\_VALUE() function and, as can be deduced from the name, returns the value from the last row of the dataset or the partition to which it is applied.
5. **Nth** **Value** :-  Finally, the NTH\_VALUE() function is very similar to both the FIRST\_VALUE() and the LAST\_VALUE(). The difference is that while the other functions output the value of either the first or the last row of a window, the NTH\_VALUE() allows the user to define which value from the order should be assigned to other rows. This function takes an additional parameter denoting which value should be returned.

**Advance Window Syntax**

1. Frame Specifications
2. EXCLUDE clause
3. FILTER clause
4. Window chaining



**Normalization**

* 1NF. Each cell should contain only one value, Primary Key,&#x20;
* 2NF. Remove partial dependency
* 3NF. transit dependency - any column which is dependent on non-primary key



**NOTES :-**&#x20;

* **Drop** will delete the table with data itself but **truncate** will only remove the data, but not the table itself (structure).
* **Delete** is used to remove the record from the table, if you don't use where clause all data from table will be removed and can be rolled back as it can’t be done in truncate.&#x20;
* Where clause is used for filtering data and can be used in select,update, delete statement, with following operator : <> not equal, in, like/not like, between

**Exception Handling**&#x20;

In MySQL, the block of SQL statements is enclosed in a Procedure. The procedure usually consists of DECLARE section and EXECUTION section.

1. DECLARE section usually consists of all variables that are locally used by the procedure for storing and processing the column values.&#x20;
2. BEGIN section is the actual execution block in real-time, which executes SQL statements, and uses a local variable.
3. A procedure when executing the block of SQL statements serially, it is quite difficult to handle an error that occurred by one of the SQL queries in the procedure.
4. Known error occurs when the SQL statements violate the properties of constraints, data types, clauses, and variables declared in the procedure
5. Exception handlers are two types:
   1. CONTINUE handler: This handler will continue to execute the procedure block of SQL statements by skipping the error-prone SQL statements in it
   2. EXIT handler: This handler will continue to execute the procedure block of SQL statements by skipping the error-prone SQL statements in it

**Query Optimization**

Optimization is a technique used while writing SQL statements to increase performance during execution. The Query performance is measured in terms of usage/cost of system resources including CPU and RAM memory. When a Query is perfectly tuned, it returns maximum throughput with minimal consumption of system resources.

Optimization of Queries is essential when developing the SQL statements otherwise it leads to: -

* Long running queries
* RAM outbound errors
* Hard disk swap – memory issue
* High Input/output network in MySQL engine
* CPU consumption goes high and leads to a database crash

**Functions and Procedure**&#x20;

* Functions and procedures are advanced features to handle the SQL statements in a procedural language.
* Since functions and procedures are named objects, they can be stored in the database
* They can be invoked at any time and are executed in the database without referring to actual SQL statements written in them
* Improves overall database design and database security.

_A function has two sections of declaration:_

1. declaration\_section: - This section enables users to define local variables with their respective data types
2. executable\_section: - This section is used for writing the program logic on DML statements or SELECT queries to store the output values that are declared above
3. Return: - RETURN keyword finally returns the value from the function execution&#x20;
4. Function is invoked in SQL query statement.
5. Function cannot be written with DML statements when it is used in SELECT queries.&#x20;

**Trigger**&#x20;

A trigger is a stored program in a database that automatically responds to an event of DML operations done by inserting, updating, or deleting. A trigger is nothing but an auditor of events happening across all database tables. Triggers are written to be executed in response to any of the DDL or DML events.&#x20;

Triggers could be defined on the table, view, schema, or database with which the event is associated Triggers ensure referential integrity and effectively work along with other referential integrity constraints: Primary key and foreign key.



#### SQL Code

<pre class="language-sql"><code class="lang-sql">Create database &#x3C;database_name>;
Use &#x3C;database_name>;
Create table &#x3C;table_name> (column1 int, column var, …..)
Insert into &#x3C;table_name> values (1, ‘test’), ….
Describe table_name;


// Alter

alter table &#x3C;table_name> rename column &#x3C;column1> to &#x3C;column2>;  -- renaming column name
alter table &#x3C;table_name>  add column &#x3C;column1> varchar(10); --  adding new column 
alter table &#x3C;table_name>  drop column &#x3C;column1>;  -- dropping column from table
alter table &#x3C;table_name> drop primary key; -- Dropping primary key
alter table &#x3C;table_name>  modify sname char(20);  --  changing datatype
alter table &#x3C;table_name>  add column age int after sname; -- adding after specific column
alter table &#x3C;table_name>  add column rollno int first;  -- add column at first in table 
alter table emp add constraint c1 primary key(rollno);
rename table &#x3C;table_name1> to &#x3C;table_name2>
alter table &#x3C;table_name> add primary key &#x3C;column_name>

// Primary Key 

create table dept1 (deptid int primary key, dename varchar(20) not null);
create table emp1 (empid int unique not null,
                   deptid int,
                   age int check( age > 17 and age &#x3C; 60),
                   email char(10) default 'NA',
                   foreign key(deptid) references dept1(deptid));

create table emp3 (empid int unique not null,
                   deptid int,
                   age int check( age > 17 and age &#x3C; 60),
                   email char(10) default 'NA',
                   foreign key(deptid) references dept1(deptid)
                   on update cascade on delete set null);
// Update

update &#x3C;table_name> set column_name=4 where column_name=3;
update &#x3C;table_name> set column_name=null  where column_name=3;
update &#x3C;table_name> set column_name1=null and column_name1=5  where column_name=3;

//Select 

select * from employee where dept_id in (50,80) and salary > 66666;
select * from employee where job_id in ('ADMIN','EXE') and salary > 66666;
select * from employee where f_name like '%a';
                                    like '%a__'
                                    like "%\%_"  --  suppress the predefined meaning and treating it like literal
                                    like 'N%'
                                    like 'm%m'

select  department_id , sum(salary) from employees  group by department_id having count(*)>30 ;
select dept_id, count(*) from emp group by dep_id, job_id with rollup; -- summary creation
select hire_date, count(hire_date)  from employees  group by hire_date  having  count(hire_date) > 1;

// Single Row Function

select upper(first_name), lower(last_name) from employee;
select * from lik where lower(name)='namrata';
select trim('   ss  ');
select trim('   ss  ff ');
select trim('n' from  'nnnjss');
select replace('BB JJ',' ','');
select substr('Namrita',4,2);  -- 4 is starting position 
select instr('ksdsds', 's');  -- position number
select substr('ASW SWS', 1, instr('ASW SWS',' ')), 
          substr('ASW SWS', instr('ASW SWS',' '));


// Multi Row Functions 

select concat(fnam,' ', lname)output from employees;
select right('ssddesds', 2);
select left('sdrerf',3);
select lpad('abcdefghij', 10, '*');
select rpad('zcdzc', 10, '*');
select round(89.89,-1);
select round(89.89);
select round(89.89,-1);
select round(89.89),floor(89.59),ceil(34.23),truncate(89.79,1);
select substr('ASW$SWS', 1, instr('ASW$SWS','$')-1);
select trim(substr('ASW SWS', 1, instr('ASW SWS',' ')));
select ifnull(343,54);
select ifnull(null,665);
select ifnull(null,null);  -- ifnull() takes only two parameter
select coalesce(null, null, 4); -- works on more than two parameter
select coalesce(null, 5, 4); -- works on more than two parameter | pick first value after null
select coalesce(null, 4, 5); -- works on more than two parameter

// CASE

case
when job_id like  '%clerk%'  then 'clerk'
when job_id like  '%mgr%' or job_id like  '%managr%'  then 'Manager'
else 'others'
end cc

<strong>// Windows Function
</strong>
select employee_id, salary, row_number() over(order by salary desc) from employees;
select employee_id, salary, rank() over(order by salary desc) from employees;
select employee_id, salary, dense_rank() over(order by salary desc) from employees;
select  *  row_number() over(partition by cust_id order by complaint_date desc) from complaints;
select employee_id ,salary, ntile(5) over(order by salary desc) from employees;

select salary, lag(salary) over() from employees;
select salary, lead(salary) over() from employees;

// View

create view V1 as 
select employee_id, salary, commission_pct 
from employees;

alter table V1 add column s int;                   
                   
</code></pre>





