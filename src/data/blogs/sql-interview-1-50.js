export const sql_interview_1_50 = {
  slug: 'sql-interview-1-50',
  title: 'SQL ইন্টারভিউ প্রশ্ন (Beginner → Intermediate) — ৫০টি প্রশ্ন উদাহরণসহ',
  description: 'Join, aggregation, index, transaction, window function এবং schema design নিয়ে ৫০টি গুরুত্বপূর্ণ SQL ইন্টারভিউ প্রশ্ন — প্রতিটির সাথে query example।',
  date: 'জুলাই ২০২৬',
  category: 'SQL ইন্টারভিউ প্রশ্ন',
  readTime: 30,
  content: `
    <h3>SQL Fundamentals (Q1–Q15)</h3>

    <p><strong>Q1. SQL-এর DDL, DML, DCL, এবং TCL-এর মধ্যে পার্থক্য কী?</strong><br/>
    DDL (Data Definition Language: CREATE, ALTER, DROP) schema structure define করে। DML (Data Manipulation Language: SELECT, INSERT, UPDATE, DELETE) data manipulate করে। DCL (Data Control Language: GRANT, REVOKE) permission manage করে। TCL (Transaction Control Language: COMMIT, ROLLBACK, SAVEPOINT) transaction manage করে।
    <pre><code>CREATE TABLE orders (...);   -- DDL
INSERT INTO orders VALUES (...);   -- DML
GRANT SELECT ON orders TO analyst;   -- DCL
COMMIT;   -- TCL</code></pre></p>

    <p><strong>Q2. WHERE এবং HAVING-এর মধ্যে পার্থক্য কী?</strong><br/>
    WHERE grouping/aggregation-এর আগে individual row filter করে। HAVING GROUP BY-এর পরে group filter করে — WHERE-এ COUNT()-এর মতো aggregate function ব্যবহার করা যায় না, কিন্তু HAVING-এ যায়।
    <pre><code>SELECT department, COUNT(*) FROM employees
WHERE salary > 50000
GROUP BY department
HAVING COUNT(*) > 5;</code></pre></p>

    <p><strong>Q3. DELETE, TRUNCATE, এবং DROP-এর মধ্যে পার্থক্য কী?</strong><br/>
    DELETE একবারে একটা করে row সরায় (WHERE support করে, logged হয়, trigger fire হয়, rollback করা যায়)। TRUNCATE একবারে সব row সরায় (দ্রুত, minimal logging, auto-increment reset করে, WHERE নেই)। DROP পুরো table structure permanently সরিয়ে দেয়।
    <pre><code>DELETE FROM orders WHERE status = 'cancelled';
TRUNCATE TABLE orders;
DROP TABLE orders;</code></pre></p>

    <p><strong>Q4. INNER JOIN, LEFT JOIN, RIGHT JOIN, এবং FULL OUTER JOIN explain করুন।</strong><br/>
    INNER JOIN দুই table-এই matching row-ই return করে। LEFT JOIN left table-এর সব row plus right-এর match return করে (match না থাকলে NULL)। RIGHT JOIN এর mirror। FULL OUTER JOIN দুই দিকেরই সব row return করে, match না থাকলে NULL সহ।
    <pre><code>SELECT c.name, o.order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;
-- প্রতিটা customer return করে, এমনকি যাদের কোনো order নেই (order_id = NULL)</code></pre></p>

    <p><strong>Q5. Self join কী এবং কখন ব্যবহার করবেন?</strong><br/>
    একটা self join একটা table-কে নিজের সাথেই join করে — একই table-এর row compare করার জন্য useful, যেমন একই employees table-এ থাকা employee-দের তাদের manager-এর সাথে match করা।
    <pre><code>SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;</code></pre></p>

    <p><strong>Q6. UNION এবং UNION ALL-এর মধ্যে পার্থক্য কী?</strong><br/>
    UNION result set combine করে duplicate row সরিয়ে দেয় (sort/dedup step দরকার, ধীর)। UNION ALL সব row রেখে দেয় duplicate সহ (দ্রুত, dedup overhead নেই)।
    <pre><code>SELECT city FROM customers
UNION ALL
SELECT city FROM suppliers;   -- duplicate city রেখে দেয়, UNION-এর চেয়ে দ্রুত</code></pre></p>

    <p><strong>Q7. Primary key বনাম foreign key কী?</strong><br/>
    Primary key নিজের table-এ প্রতিটা row-কে uniquely identify করে (duplicate নেই, NULL নেই)। Foreign key একটা table-এর column যা অন্য একটা table-এর primary key-কে reference করে, তাদের মধ্যে referential integrity বজায় রাখে।
    <pre><code>CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(id)   -- foreign key
);</code></pre></p>

    <p><strong>Q8. NULL value কী এবং comparison-এ এটা কীভাবে behave করে?</strong><br/>
    NULL unknown/missing data represent করে — এটা অন্য NULL-সহ কারো সমান নয়। = NULL সবসময় unknown evaluate হয়, তাই check করতে IS NULL / IS NOT NULL ব্যবহার করতে হবে।
    <pre><code>SELECT * FROM employees WHERE manager_id IS NULL;   -- সঠিক
SELECT * FROM employees WHERE manager_id = NULL;    -- ভুল, কিছুই return করে না</code></pre></p>

    <p><strong>Q9. COUNT(*), COUNT(column), এবং COUNT(DISTINCT column)-এর মধ্যে পার্থক্য কী?</strong><br/>
    COUNT(*) NULL-সহ সব row count করে। COUNT(column) সেই column-এর non-NULL value count করে। COUNT(DISTINCT column) unique non-NULL value count করে।
    <pre><code>-- মোট 5 row; email column-এ 1টা NULL এবং 1টা duplicate value
SELECT COUNT(*), COUNT(email), COUNT(DISTINCT email) FROM users;
-- return করে 5, 4, 3</code></pre></p>

    <p><strong>Q10. GROUP BY explain করুন এবং এটা aggregate function-এর সাথে কীভাবে কাজ করে।</strong><br/>
    GROUP BY নির্দিষ্ট column-এ একই value থাকা row-কে একটা summary row-এ collapse করে প্রতি group-এ, aggregate function (SUM, AVG, COUNT, MAX, MIN)-কে পুরো table-এর বদলে প্রতি group-এ calculate করতে দেয়।
    <pre><code>SELECT department, AVG(salary) FROM employees GROUP BY department;
-- প্রতি department-এ একটা average-salary row</code></pre></p>

    <p><strong>Q11. Subquery কী? একটা correlated subquery-এর উদাহরণ দিন।</strong><br/>
    Subquery হলো একটা query-এর ভেতরে nested আরেকটা query। Correlated subquery outer query-র একটা column reference করে, তাই এটা প্রতিটা outer row-এর জন্য একবার করে re-execute হয় (independent subquery শুধু একবারই run হয়)।
    <pre><code>SELECT e.name FROM employees e
WHERE salary > (
  SELECT AVG(salary) FROM employees WHERE department = e.department
);
-- correlated: inner AVG outer row-এর department-এর উপর নির্ভরশীল</code></pre></p>

    <p><strong>Q12. Subquery এবং JOIN-এর মধ্যে পার্থক্য কী?</strong><br/>
    দুটোই related data retrieve করে, কিন্তু JOIN দুই table-এর column একটা result set-এ combine করে, যেখানে subquery সাধারণত filtering-এর জন্য একটা single value/list return করে। Data combine করার জন্য JOIN সাধারণত বেশি efficient ও readable।
    <pre><code>-- subquery version
SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country='USA');
-- join version (প্রায়ই দ্রুত, এবং customer column-ও select করতে দেয়)
SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.country='USA';</code></pre></p>

    <p><strong>Q13. Common Table Expression (CTE) কী? কেন ব্যবহার করবেন?</strong><br/>
    CTE (WITH clause) একটা নামযুক্ত temporary result set define করে যা একটা single query-এর মধ্যে reference করা যায় — গভীরভাবে nested subquery-র চেয়ে readability বাড়ায় এবং recursive query সম্ভব করে।
    <pre><code>WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000
)
SELECT department, COUNT(*) FROM high_earners GROUP BY department;</code></pre></p>

    <p><strong>Q14. Window function কী? একটা উদাহরণ দিন।</strong><br/>
    Window function (OVER clause) GROUP BY-র মতো row collapse না করেই related row-এর একটা set জুড়ে calculation করে — প্রতিটা input row নিজের output row এবং computed window value দুটোই রাখে।
    <pre><code>SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
-- row merge না করেই প্রতি department-এর employee-দের salary অনুযায়ী rank করে</code></pre></p>

    <p><strong>Q15. RANK(), DENSE_RANK(), এবং ROW_NUMBER()-এর মধ্যে পার্থক্য কী?</strong><br/>
    ROW_NUMBER() প্রতিটা row-কে একটা unique sequential number দেয় (tie নেই)। RANK() tie-এ একই rank দেয় কিন্তু পরের number skip করে (1,2,2,4)। DENSE_RANK() tie-তে একই rank দেয় skip না করে (1,2,2,3)।<br/>
    উদাহরণ: score 90, 90, 80-এর জন্য — ROW_NUMBER দেয় 1,2,3; RANK দেয় 1,1,3; DENSE_RANK দেয় 1,1,2।</p>

    <h3>Joins, Aggregation &amp; Query Design (Q16–Q30)</h3>

    <p><strong>Q16. Database normalization কী? সংক্ষেপে 1NF, 2NF, 3NF explain করুন।</strong><br/>
    Normalization redundancy কমাতে এবং update anomaly এড়াতে data organize করে। 1NF: প্রতিটা column atomic value রাখে, কোনো repeating group নেই। 2NF: 1NF plus প্রতিটা non-key column পুরো primary key-এর উপর নির্ভরশীল। 3NF: 2NF plus কোনো non-key column অন্য একটা non-key column-এর উপর নির্ভরশীল নয়।<br/>
    উদাহরণ: প্রতিটা order-এ customer_name/address repeat করা একটা single "orders" table-কে আলাদা "customers" এবং "orders" table-এ ভাগ করা 3NF-এর দিকে যাওয়া।</p>

    <p><strong>Q17. Denormalization কী এবং কখন এটা ব্যবহার করবেন?</strong><br/>
    Denormalization ইচ্ছাকৃতভাবে redundancy যোগ করে (যেমন orders table-এ customer-এর নাম duplicate করা) দরকারি JOIN কমাতে, storage/update complexity-র বিনিময়ে read performance বাড়াতে — analytics এবং data warehouse-এ common।<br/>
    উদাহরণ: orders + customers + products আগে থেকেই join করা একটা reporting table, যাতে dashboard query তিনটা JOIN-এর বদলে একটা table hit করে।</p>

    <p><strong>Q18. Index কী এবং এটা query কীভাবে দ্রুত করে?</strong><br/>
    Index হলো একটা আলাদা data structure (সাধারণত B-tree) যা column value ও তাদের row location-এর একটা sorted reference রাখে, পুরো table scan করার বদলে দ্রুত lookup দিয়ে matching row খুঁজতে দেয়।
    <pre><code>CREATE INDEX idx_customer_email ON customers(email);
-- WHERE email = 'x@example.com' filter করা একটা query এখন index ব্যবহার করতে পারে</code></pre></p>

    <p><strong>Q19. অনেক বেশি index যোগ করার trade-off কী?</strong><br/>
    Index SELECT/WHERE/JOIN lookup দ্রুত করে কিন্তু INSERT/UPDATE/DELETE ধীর করে দেয় (প্রতিটা index-ও update হতে হয়) এবং extra disk space নেয়।<br/>
    উদাহরণ: একটা high-write logging system-এ 15টা index থাকা একটা table প্রতিটা INSERT-কে ভালো করে বেছে নেওয়া 2টা index থাকা একই table-এর চেয়ে লক্ষণীয়ভাবে ধীর করে দিতে পারে।</p>

    <p><strong>Q20. Composite index কী এবং column order কি গুরুত্বপূর্ণ?</strong><br/>
    একটা composite index একাধিক column জুড়ে থাকে। Order গুরুত্বপূর্ণ — index শুধু সেই query-তে usable যা indexed column-গুলোর left-to-right prefix-এ filter করে।
    <pre><code>CREATE INDEX idx_name_dept ON employees(last_name, department);
-- সাহায্য করে: WHERE last_name = 'Smith'
-- সাহায্য করে: WHERE last_name = 'Smith' AND department = 'Sales'
-- সাহায্য করে না: শুধু WHERE department = 'Sales'</code></pre></p>

    <p><strong>Q21. Transaction কী এবং ACID কী বোঝায়?</strong><br/>
    Transaction হলো একটা group of operation যা একটা single all-or-nothing unit হিসেবে execute হয়। ACID: Atomicity (সব-নাহলে-কিছুই-না), Consistency (আগে/পরে valid state), Isolation (concurrent transaction একে অপরকে প্রভাবিত করে না), Durability (commit হওয়া change crash-এও টিকে থাকে)।
    <pre><code>BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- দ্বিতীয় UPDATE fail করলে পুরো transaction rollback হয়ে যায় যাতে টাকা হারিয়ে না যায়</code></pre></p>

    <p><strong>Q22. Transaction isolation level কী কী?</strong><br/>
    READ UNCOMMITTED (অন্য transaction-এর uncommitted change দেখতে পারে — dirty read), READ COMMITTED (শুধু committed data দেখে, common default), REPEATABLE READ (একটা transaction-এর ভেতরে একই query একই row return করে), SERIALIZABLE (সবচেয়ে strict — transaction একটার পর একটা run হওয়ার মতো behave করে)।<br/>
    উদাহরণ: READ UNCOMMITTED-এ, Transaction A এমন একটা balance update পড়তে পারে যা Transaction B পরে rollback করে দেয় — একটা "dirty read"।</p>

    <p><strong>Q23. Database-এ deadlock কী?</strong><br/>
    দুই বা তার বেশি transaction যখন একে অপরের দরকারি lock ধরে রাখে, তখন deadlock হয়, ফলে কেউই এগোতে পারে না — database এটা detect করে এবং সাধারণত cycle ভাঙতে একটা transaction (the "victim") kill করে।<br/>
    উদাহরণ: Transaction 1 row A lock করে row B-এর জন্য wait করে; Transaction 2 row B lock করে row A-এর জন্য wait করে — কেউই চালিয়ে যেতে পারে না।</p>

    <p><strong>Q24. SQL-এ view কী?</strong><br/>
    View হলো একটা save করা, নামযুক্ত SELECT query যা একটা virtual table-এর মতো behave করে — এটা নিজে data store করে না (সাধারণত) কিন্তু complex/repeated query সহজ করে এবং user কোন column/row দেখবে তা restrict করতে পারে।
    <pre><code>CREATE VIEW active_customers AS
SELECT id, name, email FROM customers WHERE status = 'active';
SELECT * FROM active_customers;   -- একটা সাধারণ table-এর মতো view query করুন</code></pre></p>

    <p><strong>Q25. Stored procedure কী এবং কেন ব্যবহার করবেন?</strong><br/>
    Stored procedure হলো database-এ store করা একটা precompiled, নামযুক্ত SQL block (optional parameter এবং control-flow logic সহ), repeated network round-trip কমায় এবং business logic-কে data-এর কাছে centralize করে।
    <pre><code>CREATE PROCEDURE GetHighEarners(IN min_salary DECIMAL)
BEGIN
  SELECT * FROM employees WHERE salary > min_salary;
END;
CALL GetHighEarners(100000);</code></pre></p>

    <p><strong>Q26. Clustered এবং non-clustered index-এর মধ্যে পার্থক্য কী?</strong><br/>
    Clustered index disk-এ data আসলে কোন physical order-এ store হবে তা নির্ধারণ করে (প্রতি table-এ একটাই, সাধারণত primary key)। Non-clustered index একটা আলাদা structure যা আসল row-এর দিকে pointer রাখে (একটা table-এ অনেকগুলো থাকতে পারে)।<br/>
    উদাহরণ: SQL Server-এ, primary key default-ভাবে একটা clustered index তৈরি করে, সেই key দিয়ে table-কে physically sort করে।</p>

    <p><strong>Q27. একটা table-এ duplicate row কীভাবে খুঁজবেন?</strong><br/>
    "Duplicate" নির্ধারণকারী column-গুলো দিয়ে GROUP BY করুন এবং HAVING COUNT(*) &gt; 1 দিয়ে filter করুন।
    <pre><code>SELECT email, COUNT(*) FROM users
GROUP BY email
HAVING COUNT(*) > 1;</code></pre></p>

    <p><strong>Q28. একটা table-এ second-highest salary কীভাবে খুঁজবেন?</strong><br/>
    একাধিক approach: max বাদ দিয়ে একটা subquery, ORDER BY-এর সাথে LIMIT/OFFSET, অথবা DENSE_RANK()।
    <pre><code>SELECT MAX(salary) FROM employees
WHERE salary &lt; (SELECT MAX(salary) FROM employees);
-- অথবা: SELECT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;</code></pre></p>

    <p><strong>Q29. WHERE কেন COUNT(*)-এর মতো aggregate-এ filter করতে পারে না, কিন্তু HAVING পারে?</strong><br/>
    WHERE GROUP BY/aggregation হওয়ার আগে evaluate হয়, তাই execution-এর সেই পর্যায়ে aggregate value এখনো তৈরিই হয়নি — HAVING aggregation-এর পরে run হয়, যখন aggregate result filter করার জন্য available।<br/>
    উদাহরণ: SELECT dept, COUNT(*) FROM emp WHERE COUNT(*) &gt; 5 GROUP BY dept invalid; এর বদলে HAVING COUNT(*) &gt; 5 ব্যবহার করতে হবে।</p>

    <p><strong>Q30. ON DELETE CASCADE কী এবং এর risk কী?</strong><br/>
    ON DELETE CASCADE reference করা parent row delete হলে automatically child row-ও delete করে দেয়, manual cleanup ছাড়াই referential integrity বজায় রাখে — কিন্তু risk হলো ভালোভাবে scope না করলে chain reaction-এ অনেক related data accidentally delete হয়ে যেতে পারে।
    <pre><code>CREATE TABLE orders (
  customer_id INT REFERENCES customers(id) ON DELETE CASCADE
);
-- একটা customer delete করলে তার সব order automatically delete হয়ে যায়</code></pre></p>

    <h3>Advanced &amp; Intermediate Topics (Q31–Q50)</h3>

    <p><strong>Q31. Scalar subquery এবং table subquery-এর মধ্যে পার্থক্য কী?</strong><br/>
    Scalar subquery exactly একটা row ও একটা column return করে (যেখানেই একটা single value দরকার সেখানে ব্যবহারযোগ্য)। Table subquery একাধিক row/column return করে (FROM-এ বা IN/EXISTS-এর সাথে ব্যবহারযোগ্য)।
    <pre><code>SELECT name, (SELECT AVG(salary) FROM employees) AS company_avg FROM employees;
-- একটা computed column হিসেবে ব্যবহৃত scalar subquery</code></pre></p>

    <p><strong>Q32. EXISTS clause কী এবং IN থেকে এটা কীভাবে আলাদা?</strong><br/>
    EXISTS check করে subquery কোনো row return করে কিনা (প্রথম match-এই থেমে যায়, বড় subquery-তে প্রায়ই দ্রুত, NULL আরও predictably handle করে), যেখানে IN একটা value list-এর বিরুদ্ধে membership check করে (list-এ NULL থাকলে অপ্রত্যাশিতভাবে behave করতে পারে)।
    <pre><code>SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
-- যাদের অন্তত একটা order আছে সেই customer-দের return করে</code></pre></p>

    <p><strong>Q33. NULL value থাকা column-এ JOIN করলে কী হয়?</strong><br/>
    NULL অন্য NULL-সহ কারো সমান নয়, তাই join column-এ NULL থাকা row একটা standard equality JOIN-এ কখনো match করবে না — explicitly handle না করলে সেগুলো result থেকে বাদ পড়ে যায়।<br/>
    উদাহরণ: a.customer_id = b.customer_id দুই দিকেই NULL হলেও match করবে না।</p>

    <p><strong>Q34. SQL-এর logical query execution order কী?</strong><br/>
    আপনি প্রথমে SELECT লিখলেও, SQL logically process করে: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT। এই কারণেই WHERE-এ SELECT alias reference করা যায় না, কিন্তু ORDER BY-এ প্রায়ই যায়।<br/>
    উদাহরণ: SELECT salary*1.1 AS new_salary FROM emp WHERE new_salary &gt; 50000 অনেক database-এ fail করে কারণ alias তৈরি হওয়ার আগেই WHERE run হয়।</p>

    <p><strong>Q35. EXPLAIN plan কী এবং কেন ব্যবহার করবেন?</strong><br/>
    EXPLAIN (বা EXPLAIN ANALYZE) একটা query-এর জন্য database-এর execution plan দেখায় — কোন index ব্যবহার হচ্ছে, join order, estimated row count, এবং cost — ধীর query diagnose করতে এবং কোথায় index সাহায্য করতে পারে সিদ্ধান্ত নিতে সাহায্য করে।<br/>
    উদাহরণ: EXPLAIN SELECT * FROM orders WHERE customer_id = 5; হয়তো একটা full table scan দেখাবে index scan-এর বদলে, customer_id-তে একটা missing index-এর ইঙ্গিত দিয়ে।</p>

    <p><strong>Q36. Temporary table এবং CTE-এর মধ্যে পার্থক্য কী?</strong><br/>
    Temporary table physically materialize হয়, session/transaction জুড়ে থাকে, index করা যায়, এবং একাধিক query জুড়ে reuse করা যায়। CTE শুধু একটা single statement-এ scope করা নামযুক্ত subquery — materialize/index হয় না (বেশিরভাগ database-এ) এবং সেই query run হওয়ার পরপরই discard হয়ে যায়।
    <pre><code>CREATE TEMP TABLE recent_orders AS
SELECT * FROM orders WHERE order_date > '2026-01-01';
-- CTE-এর বিপরীতে, একই session-এ একাধিকবার query করা যায়</code></pre></p>

    <p><strong>Q37. N+1 query problem কী এবং এটা কীভাবে এড়াবেন?</strong><br/>
    N+1 problem হয় যখন code N-টা parent record fetch করে, তারপর প্রতিটার related child record-এর জন্য আলাদা query চালায় (মোট 1 + N query) — একটা JOIN বা batched query-তে সবকিছু fetch করার বদলে — একটা common ORM performance pitfall।<br/>
    উদাহরণ: 100টা blog post fetch করে, তারপর loop করে প্রতিটা post-এর জন্য comments query চালানো (101টা query) একটা WHERE post_id IN (...) query-র বদলে।</p>

    <p><strong>Q38. Self-referencing foreign key কী?</strong><br/>
    একটা table-এর foreign key column যা সেই একই table-এর primary key-কে reference করে — org chart বা category tree-র মতো hierarchical data-তে common।
    <pre><code>CREATE TABLE categories (
  id INT PRIMARY KEY,
  parent_id INT REFERENCES categories(id)
);</code></pre></p>

    <p><strong>Q39. UNION এবং JOIN-এর মধ্যে পার্থক্য কী?</strong><br/>
    JOIN একটা related key-এর ভিত্তিতে দুই table-এর column পাশাপাশি combine করে (বেশি column, একই বা কম row)। UNION দুটো query-র row একটার উপর আরেকটা stack করে (একই সংখ্যক column, বেশি row) — union করা query-গুলোর column সংখ্যা ও compatible type মিলতে হবে।<br/>
    উদাহরণ: UNION "2026 sales" এবং "2025 sales"-কে একটা লম্বা list-এ combine করে; JOIN "orders" এবং "customers"-কে একটা চওড়া table-এ combine করে।</p>

    <p><strong>Q40. SQL-এ pagination কীভাবে handle করবেন?</strong><br/>
    ORDER BY-এর পরে LIMIT এবং OFFSET ব্যবহার করে result-এর একটা নির্দিষ্ট page নিন — যদিও খুব বড় table-এ OFFSET ধীর হয়ে যায় কারণ database-কে এখনো আগের row scan/skip করতে হয়; keyset pagination (WHERE id &gt; last_seen_id) ভালো scale করে।
    <pre><code>SELECT * FROM products ORDER BY id LIMIT 20 OFFSET 40;   -- page 3, প্রতি page-এ 20</code></pre></p>

    <p><strong>Q41. Scalar function এবং aggregate function-এর মধ্যে পার্থক্য কী?</strong><br/>
    একটা scalar function (UPPER, ROUND, LENGTH) প্রতিটা individual row-এ কাজ করে এবং একটা value return করে। একটা aggregate function (SUM, AVG, COUNT, MAX) একাধিক row জুড়ে কাজ করে এবং প্রতি group-এ একটা summarized value return করে।<br/>
    উদাহরণ: SELECT UPPER(name) (scalar, প্রতি row-এ একটা result) বনাম SELECT AVG(salary) (aggregate, প্রতি group-এ একটা result)।</p>

    <p><strong>Q42. COALESCE function কীসের জন্য ব্যবহার হয়?</strong><br/>
    COALESCE(a, b, c, ...) list-এর প্রথম non-NULL value return করে — কোনো column NULL হতে পারলে একটা default value substitute করার জন্য commonly ব্যবহৃত।
    <pre><code>SELECT name, COALESCE(phone, 'No phone on file') AS contact FROM customers;</code></pre></p>

    <p><strong>Q43. CHECK constraint কী?</strong><br/>
    একটা CHECK constraint database level-এ column value-র উপর একটা boolean condition enforce করে, সেটা violate করা যেকোনো INSERT/UPDATE reject করে — application-level validation থেকে independent।
    <pre><code>CREATE TABLE products (
  price DECIMAL CHECK (price > 0)
);
-- INSERT INTO products (price) VALUES (-5); reject হয়ে যাবে</code></pre></p>

    <p><strong>Q44. UNIQUE constraint এবং PRIMARY KEY-এর মধ্যে পার্থক্য কী?</strong><br/>
    দুটোই uniqueness enforce করে, কিন্তু একটা table-এ শুধু একটাই PRIMARY KEY থাকতে পারে (যেটা NOT NULL-ও বোঝায়), যেখানে একাধিক UNIQUE constraint থাকতে পারে — এবং বেশিরভাগ database-এ UNIQUE column একটা NULL value allow করতে পারে।
    <pre><code>CREATE TABLE users (
  id INT PRIMARY KEY,
  email VARCHAR UNIQUE   -- unique, কিন্তু theoretically একটা NULL allow করতে পারে
);</code></pre></p>

    <p><strong>Q45. একটা ধীর SQL query কীভাবে optimize করবেন?</strong><br/>
    Full table scan-এর জন্য EXPLAIN plan check করুন, WHERE/JOIN/ORDER BY-তে ব্যবহৃত column-এ index যোগ করুন, SELECT * এড়িয়ে চলুন (শুধু দরকারি column fetch করুন), WHERE-এ indexed column-এর উপর function ব্যবহার এড়িয়ে চলুন (index usage ভেঙে দেয়), এবং যেখানে সম্ভব correlated subquery-কে JOIN হিসেবে rewrite করুন।
    <pre><code>-- order_date-এ index usage আটকে দেয়:
WHERE YEAR(order_date) = 2026
-- rewrite করুন যাতে index ব্যবহার করা যায়:
WHERE order_date >= '2026-01-01' AND order_date &lt; '2027-01-01'</code></pre></p>

    <p><strong>Q46. Natural join এবং explicit JOIN...ON-এর মধ্যে পার্থক্য কী?</strong><br/>
    NATURAL JOIN দুই table-এই একই নামের সব column-এ automatically join করে (implicit, column name coincidentally মিলে গেলে বা বদলালে risky), যেখানে JOIN...ON আপনাকে explicitly join condition specify করতে দেয় — clarity ও safety-র জন্য production code-এ strongly preferred।<br/>
    উদাহরণ: কেউ যদি matching নামের একটা unrelated column যোগ করে, NATURAL JOIN silently ভেঙে যায়।</p>

    <p><strong>Q47. Window frame (ROWS BETWEEN) কী?</strong><br/>
    একটা window frame define করে একটা partition-এর মধ্যে কোন row-এর subset একটা window function বিবেচনা করবে, যেমন running total বা moving average — ROWS BETWEEN বা RANGE BETWEEN দিয়ে specify করা হয়।
    <pre><code>SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_sum_3
FROM orders;
-- একটা 3-row moving sum calculate করে</code></pre></p>

    <p><strong>Q48. LEFT JOIN এবং LEFT JOIN ... WHERE right.col IS NULL-এর মধ্যে পার্থক্য কী?</strong><br/>
    একটা সাধারণ LEFT JOIN সব left row plus match return করে (unmatched হলে NULL)। WHERE right_table.col IS NULL যোগ করলে শুধু সেই left row-গুলোতে filter হয় যাদের কোনো match-ই নেই — "কোনো related record নেই" খুঁজে পাওয়ার একটা common pattern।
    <pre><code>SELECT c.* FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;   -- যারা কখনো order করেনি এমন customer</code></pre></p>

    <p><strong>Q49. Materialized view কী এবং সাধারণ view থেকে এটা কীভাবে আলাদা?</strong><br/>
    একটা সাধারণ view query করা হলেই তার underlying query আবার run করে (সবসময় fresh, কিন্তু ধীর হতে পারে)। একটা materialized view query result physically store করে এবং manually/periodically refresh করতে হয় — পড়তে অনেক দ্রুত কিন্তু refresh না হওয়া পর্যন্ত stale data দিতে পারে।
    <pre><code>CREATE MATERIALIZED VIEW monthly_sales AS
SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) FROM orders GROUP BY 1;
REFRESH MATERIALIZED VIEW monthly_sales;</code></pre></p>

    <p><strong>Q50. Many-to-many relationship (যেমন students এবং courses)-এর জন্য schema কীভাবে design করবেন?</strong><br/>
    দুই related table-এর foreign key রাখা একটা junction/bridge table ব্যবহার করুন (নিজের বা একটা composite primary key সহ), কারণ একটা direct many-to-many relationship একটা single foreign key column দিয়ে represent করা যায় না।
    <pre><code>CREATE TABLE enrollments (
  student_id INT REFERENCES students(id),
  course_id INT REFERENCES courses(id),
  enrolled_date DATE,
  PRIMARY KEY (student_id, course_id)
);</code></pre></p>
  `
};
