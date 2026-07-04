export const sqlEn = [
  {
    slug: 'sql-interview-1-50',
    title: 'SQL Interview Questions (Beginner → Intermediate) — 50 Q&A with Examples',
    description: '50 must-know SQL interview questions covering joins, aggregation, indexes, transactions, window functions, and schema design — each with a query example.',
    category: 'SQL Interview Questions',
    readTime: 30,
    content: `
<h3>SQL Fundamentals (Q1–Q15)</h3>

<p><strong>Q1. What is the difference between SQL's DDL, DML, DCL, and TCL?</strong><br/>
DDL (Data Definition Language: CREATE, ALTER, DROP) defines schema structure. DML (Data Manipulation Language: SELECT, INSERT, UPDATE, DELETE) manipulates data. DCL (Data Control Language: GRANT, REVOKE) manages permissions. TCL (Transaction Control Language: COMMIT, ROLLBACK, SAVEPOINT) manages transactions.
<pre><code>CREATE TABLE orders (...);   -- DDL
INSERT INTO orders VALUES (...);   -- DML
GRANT SELECT ON orders TO analyst;   -- DCL
COMMIT;   -- TCL</code></pre></p>

<p><strong>Q2. What is the difference between WHERE and HAVING?</strong><br/>
WHERE filters individual rows before grouping/aggregation. HAVING filters groups after GROUP BY — you can't use an aggregate function like COUNT() in WHERE, but you can in HAVING.
<pre><code>SELECT department, COUNT(*) FROM employees
WHERE salary > 50000
GROUP BY department
HAVING COUNT(*) > 5;</code></pre></p>

<p><strong>Q3. What's the difference between DELETE, TRUNCATE, and DROP?</strong><br/>
DELETE removes rows one at a time (supports WHERE, logged, triggers fire, rollback-able). TRUNCATE removes all rows at once (faster, minimal logging, resets auto-increment, no WHERE). DROP removes the entire table structure permanently.
<pre><code>DELETE FROM orders WHERE status = 'cancelled';
TRUNCATE TABLE orders;
DROP TABLE orders;</code></pre></p>

<p><strong>Q4. Explain INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN.</strong><br/>
INNER JOIN returns only matching rows in both tables. LEFT JOIN returns all left-table rows plus matches from the right (NULL if no match). RIGHT JOIN is the mirror. FULL OUTER JOIN returns all rows from both sides, with NULLs where there's no match.
<pre><code>SELECT c.name, o.order_id
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id;
-- returns every customer, even those with zero orders (order_id = NULL)</code></pre></p>

<p><strong>Q5. What is a self join, and when would you use one?</strong><br/>
A self join joins a table to itself — useful for comparing rows within the same table, like matching employees to their managers stored in the same table.
<pre><code>SELECT e.name AS employee, m.name AS manager
FROM employees e
JOIN employees m ON e.manager_id = m.id;</code></pre></p>

<p><strong>Q6. What is the difference between UNION and UNION ALL?</strong><br/>
UNION combines result sets and removes duplicate rows (requires a sort/dedup step, slower). UNION ALL keeps all rows including duplicates (faster, no dedup overhead).
<pre><code>SELECT city FROM customers
UNION ALL
SELECT city FROM suppliers;   -- keeps duplicate cities, faster than UNION</code></pre></p>

<p><strong>Q7. What is a primary key vs a foreign key?</strong><br/>
A primary key uniquely identifies each row in its own table (no duplicates, no NULLs). A foreign key is a column in one table that references a primary key in another, enforcing referential integrity between them.
<pre><code>CREATE TABLE orders (
  id INT PRIMARY KEY,
  customer_id INT REFERENCES customers(id)   -- foreign key
);</code></pre></p>

<p><strong>Q8. What is a NULL value, and how does it behave in comparisons?</strong><br/>
NULL represents unknown/missing data — it's not equal to anything, not even another NULL. <code>= NULL</code> always evaluates to unknown, so you must use <code>IS NULL</code> / <code>IS NOT NULL</code> to test for it.
<pre><code>SELECT * FROM employees WHERE manager_id IS NULL;   -- correct
SELECT * FROM employees WHERE manager_id = NULL;    -- WRONG, returns nothing</code></pre></p>

<p><strong>Q9. What is the difference between COUNT(*), COUNT(column), and COUNT(DISTINCT column)?</strong><br/>
COUNT(*) counts all rows including NULLs. COUNT(column) counts non-NULL values in that column. COUNT(DISTINCT column) counts unique non-NULL values.
<pre><code>-- 5 rows total; email column has 1 NULL and 1 duplicate value
SELECT COUNT(*), COUNT(email), COUNT(DISTINCT email) FROM users;
-- returns 5, 4, 3</code></pre></p>

<p><strong>Q10. Explain GROUP BY and how it works with aggregate functions.</strong><br/>
GROUP BY collapses rows sharing the same value(s) in specified columns into one summary row per group, letting aggregate functions (SUM, AVG, COUNT, MAX, MIN) compute per-group instead of over the whole table.
<pre><code>SELECT department, AVG(salary) FROM employees GROUP BY department;
-- one average-salary row per department</code></pre></p>

<p><strong>Q11. What is a subquery? Give an example of a correlated subquery.</strong><br/>
A subquery is a query nested inside another query. A correlated subquery references a column from the outer query, so it re-executes once per outer row (an independent subquery runs only once).
<pre><code>SELECT e.name FROM employees e
WHERE salary > (
  SELECT AVG(salary) FROM employees WHERE department = e.department
);
-- correlated: the inner AVG depends on the outer row's department</code></pre></p>

<p><strong>Q12. What is the difference between a subquery and a JOIN?</strong><br/>
Both retrieve related data, but a JOIN combines columns from both tables into one result set, while a subquery typically returns a single value/list used for filtering. JOINs are usually more efficient and readable for combining data.
<pre><code>-- subquery version
SELECT * FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE country='USA');
-- join version (often faster, and lets you select customer columns too)
SELECT o.* FROM orders o JOIN customers c ON o.customer_id = c.id WHERE c.country='USA';</code></pre></p>

<p><strong>Q13. What is a Common Table Expression (CTE)? Why use one?</strong><br/>
A CTE (WITH clause) defines a named temporary result set referenced within a single query — improving readability over deeply nested subqueries and enabling recursive queries.
<pre><code>WITH high_earners AS (
  SELECT * FROM employees WHERE salary > 100000
)
SELECT department, COUNT(*) FROM high_earners GROUP BY department;</code></pre></p>

<p><strong>Q14. What are window functions? Give an example.</strong><br/>
Window functions (OVER clause) perform calculations across a set of related rows without collapsing rows like GROUP BY does — each input row keeps its own output row plus a computed window value.
<pre><code>SELECT name, department, salary,
       RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS dept_rank
FROM employees;
-- ranks employees within each department by salary, without merging rows</code></pre></p>

<p><strong>Q15. What's the difference between RANK(), DENSE_RANK(), and ROW_NUMBER()?</strong><br/>
ROW_NUMBER() assigns a unique sequential number to each row (no ties). RANK() assigns the same rank to ties but skips subsequent numbers (1,2,2,4). DENSE_RANK() assigns the same rank to ties without skipping (1,2,2,3).<br/>
<em>Example:</em> For scores 90, 90, 80 — ROW_NUMBER gives 1,2,3; RANK gives 1,1,3; DENSE_RANK gives 1,1,2.</p>

<h3>Joins, Aggregation &amp; Query Design (Q16–Q30)</h3>

<p><strong>Q16. What is database normalization? Explain 1NF, 2NF, 3NF briefly.</strong><br/>
Normalization organizes data to reduce redundancy and avoid update anomalies. 1NF: every column holds atomic values, no repeating groups. 2NF: 1NF plus every non-key column depends on the whole primary key. 3NF: 2NF plus no non-key column depends on another non-key column.<br/>
<em>Example:</em> Splitting a single "orders" table that repeats customer_name/address for every order into separate "customers" and "orders" tables moves toward 3NF.</p>

<p><strong>Q17. What is denormalization, and when would you use it?</strong><br/>
Denormalization intentionally introduces redundancy (e.g., duplicating a customer's name into the orders table) to reduce JOINs needed, trading storage/update complexity for read performance — common in analytics and data warehouses.<br/>
<em>Example:</em> A reporting table that pre-joins orders + customers + products so a dashboard query hits one table instead of three JOINs.</p>

<p><strong>Q18. What is an index, and how does it speed up queries?</strong><br/>
An index is a separate data structure (commonly a B-tree) storing a sorted reference to column values and their row locations, letting the database find matching rows via fast lookup instead of a full table scan.
<pre><code>CREATE INDEX idx_customer_email ON customers(email);
-- a query filtering WHERE email = 'x@example.com' can now use the index</code></pre></p>

<p><strong>Q19. What are the trade-offs of adding too many indexes?</strong><br/>
Indexes speed up SELECT/WHERE/JOIN lookups but slow down INSERT/UPDATE/DELETE (every index must also be updated) and consume additional disk space.<br/>
<em>Example:</em> A table with 15 indexes on a high-write logging system can make each INSERT noticeably slower than the same table with 2 well-chosen indexes.</p>

<p><strong>Q20. What is a composite index, and does column order matter?</strong><br/>
A composite index spans multiple columns. Order matters — the index is only usable for queries filtering on a left-to-right prefix of the indexed columns.
<pre><code>CREATE INDEX idx_name_dept ON employees(last_name, department);
-- helps: WHERE last_name = 'Smith'
-- helps: WHERE last_name = 'Smith' AND department = 'Sales'
-- does NOT help: WHERE department = 'Sales' alone</code></pre></p>

<p><strong>Q21. What is a transaction, and what does ACID stand for?</strong><br/>
A transaction is a group of operations executed as a single all-or-nothing unit. ACID: Atomicity (all-or-nothing), Consistency (valid state before/after), Isolation (concurrent transactions don't interfere), Durability (committed changes survive crashes).
<pre><code>BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
COMMIT;
-- if the second UPDATE fails, the whole transaction rolls back so money isn't lost</code></pre></p>

<p><strong>Q22. What are the different transaction isolation levels?</strong><br/>
READ UNCOMMITTED (can see uncommitted changes — dirty reads), READ COMMITTED (only sees committed data, common default), REPEATABLE READ (same query returns same rows within a transaction), SERIALIZABLE (strictest — transactions behave as if run one at a time).<br/>
<em>Example:</em> Under READ UNCOMMITTED, Transaction A could read a balance update that Transaction B later rolls back — a "dirty read."</p>

<p><strong>Q23. What is a deadlock in databases?</strong><br/>
A deadlock occurs when two or more transactions each hold a lock the other needs, so neither can proceed — the database detects this and typically kills one transaction (the "victim") to break the cycle.<br/>
<em>Example:</em> Transaction 1 locks row A then waits for row B; Transaction 2 locks row B then waits for row A — neither can continue.</p>

<p><strong>Q24. What is a view in SQL?</strong><br/>
A view is a saved, named SELECT query that behaves like a virtual table — it doesn't store data itself (usually) but simplifies complex/repeated queries and can restrict which columns/rows a user sees.
<pre><code>CREATE VIEW active_customers AS
SELECT id, name, email FROM customers WHERE status = 'active';
SELECT * FROM active_customers;   -- query the view like a regular table</code></pre></p>

<p><strong>Q25. What is a stored procedure, and why use one?</strong><br/>
A stored procedure is a precompiled, named block of SQL (with optional parameters and control-flow logic) stored in the database, reducing repeated network round-trips and centralizing business logic close to the data.
<pre><code>CREATE PROCEDURE GetHighEarners(IN min_salary DECIMAL)
BEGIN
  SELECT * FROM employees WHERE salary > min_salary;
END;
CALL GetHighEarners(100000);</code></pre></p>

<p><strong>Q26. What is the difference between a clustered and a non-clustered index?</strong><br/>
A clustered index determines the actual physical order data is stored on disk (only one per table, usually the primary key). A non-clustered index is a separate structure with pointers back to the actual rows (a table can have many).<br/>
<em>Example:</em> In SQL Server, the primary key by default creates a clustered index, physically sorting the table by that key.</p>

<p><strong>Q27. How do you find duplicate rows in a table?</strong><br/>
GROUP BY the columns that define "duplicate" and filter with HAVING COUNT(*) &gt; 1.
<pre><code>SELECT email, COUNT(*) FROM users
GROUP BY email
HAVING COUNT(*) > 1;</code></pre></p>

<p><strong>Q28. How do you find the second-highest salary in a table?</strong><br/>
Several approaches: a subquery excluding the max, ORDER BY with LIMIT/OFFSET, or DENSE_RANK().
<pre><code>SELECT MAX(salary) FROM employees
WHERE salary &lt; (SELECT MAX(salary) FROM employees);
-- or: SELECT salary FROM employees ORDER BY salary DESC LIMIT 1 OFFSET 1;</code></pre></p>

<p><strong>Q29. Why can't WHERE filter on an aggregate like COUNT(*), but HAVING can?</strong><br/>
WHERE is evaluated before GROUP BY/aggregation happens, so the aggregate value doesn't exist yet at that stage of execution — HAVING runs after aggregation, when the aggregate result is available to filter on.<br/>
<em>Example:</em> <code>SELECT dept, COUNT(*) FROM emp WHERE COUNT(*) &gt; 5 GROUP BY dept</code> is invalid; you must use <code>HAVING COUNT(*) &gt; 5</code> instead.</p>

<p><strong>Q30. What is ON DELETE CASCADE, and what's the risk?</strong><br/>
ON DELETE CASCADE automatically deletes child rows when the referenced parent row is deleted, keeping referential integrity without manual cleanup — but risks accidentally deleting large amounts of related data in a chain reaction if not carefully scoped.
<pre><code>CREATE TABLE orders (
  customer_id INT REFERENCES customers(id) ON DELETE CASCADE
);
-- deleting a customer automatically deletes all of their orders too</code></pre></p>

<h3>Advanced &amp; Intermediate Topics (Q31–Q50)</h3>

<p><strong>Q31. What is the difference between a scalar subquery and a table subquery?</strong><br/>
A scalar subquery returns exactly one row and one column (usable anywhere a single value is expected). A table subquery returns multiple rows/columns (usable in FROM or with IN/EXISTS).
<pre><code>SELECT name, (SELECT AVG(salary) FROM employees) AS company_avg FROM employees;
-- scalar subquery used as a computed column</code></pre></p>

<p><strong>Q32. What is the EXISTS clause, and how is it different from IN?</strong><br/>
EXISTS checks whether a subquery returns any rows at all (stops at the first match, often faster, handles NULLs more predictably), while IN checks membership against a list of values (can behave unexpectedly if the list contains NULL).
<pre><code>SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
-- returns customers who have at least one order</code></pre></p>

<p><strong>Q33. What happens if you JOIN on a column that has NULL values?</strong><br/>
NULL never equals anything, including another NULL, so rows with NULL in the join column will never match in a standard equality JOIN — they're effectively excluded from the results unless explicitly handled.<br/>
<em>Example:</em> <code>a.customer_id = b.customer_id</code> will not match rows where either side is NULL, even if both are NULL.</p>

<p><strong>Q34. What is SQL's logical query execution order?</strong><br/>
Even though you write SELECT first, SQL logically processes: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT. This is why you can't reference a SELECT alias in WHERE, but you often can in ORDER BY.<br/>
<em>Example:</em> <code>SELECT salary*1.1 AS new_salary FROM emp WHERE new_salary &gt; 50000</code> fails in many databases because WHERE runs before the alias exists.</p>

<p><strong>Q35. What is an EXPLAIN plan, and why would you use it?</strong><br/>
EXPLAIN (or EXPLAIN ANALYZE) shows the database's execution plan for a query — which indexes it uses, join order, estimated row counts, and cost — helping diagnose slow queries and decide where an index might help.<br/>
<em>Example:</em> <code>EXPLAIN SELECT * FROM orders WHERE customer_id = 5;</code> might reveal a full table scan instead of an index scan, signaling a missing index on customer_id.</p>

<p><strong>Q36. What is the difference between a temporary table and a CTE?</strong><br/>
A temporary table is physically materialized, persists for the session/transaction, can be indexed, and reused across multiple queries. A CTE is just a named subquery scoped to a single statement — not materialized/indexed and discarded after that query runs.
<pre><code>CREATE TEMP TABLE recent_orders AS
SELECT * FROM orders WHERE order_date > '2026-01-01';
-- can be queried multiple times in the same session, unlike a CTE</code></pre></p>

<p><strong>Q37. What is the N+1 query problem, and how do you avoid it?</strong><br/>
The N+1 problem happens when code fetches N parent records, then runs a separate query for each one's related child records (1 + N queries total) instead of fetching everything in one JOIN or batched query — a common ORM performance pitfall.<br/>
<em>Example:</em> Fetching 100 blog posts, then looping and running a comments query per post (101 queries) instead of one query with <code>WHERE post_id IN (...)</code>.</p>

<p><strong>Q38. What is a self-referencing foreign key?</strong><br/>
A foreign key column in a table that references the primary key of that same table — common for hierarchical data like org charts or category trees.
<pre><code>CREATE TABLE categories (
  id INT PRIMARY KEY,
  parent_id INT REFERENCES categories(id)
);</code></pre></p>

<p><strong>Q39. What is the difference between UNION and JOIN?</strong><br/>
JOIN combines columns from two tables side-by-side based on a related key (more columns, same or fewer rows). UNION stacks rows from two queries on top of each other (same number of columns, more rows) — the queries must have matching column counts and compatible types.<br/>
<em>Example:</em> UNION combines "2026 sales" and "2025 sales" into one taller list; JOIN combines "orders" and "customers" into one wider table.</p>

<p><strong>Q40. How do you handle pagination in SQL?</strong><br/>
Use LIMIT and OFFSET after ORDER BY to retrieve a specific page of results — though OFFSET becomes slow on very large tables since the database still has to scan/skip preceding rows; keyset pagination (WHERE id &gt; last_seen_id) scales better.
<pre><code>SELECT * FROM products ORDER BY id LIMIT 20 OFFSET 40;   -- page 3, 20 per page</code></pre></p>

<p><strong>Q41. What's the difference between a scalar function and an aggregate function?</strong><br/>
A scalar function (UPPER, ROUND, LENGTH) operates on and returns a value per individual row. An aggregate function (SUM, AVG, COUNT, MAX) operates across multiple rows and returns one summarized value per group.<br/>
<em>Example:</em> <code>SELECT UPPER(name)</code> (scalar, one result per row) vs <code>SELECT AVG(salary)</code> (aggregate, one result per group).</p>

<p><strong>Q42. What is the COALESCE function used for?</strong><br/>
COALESCE(a, b, c, ...) returns the first non-NULL value in the list — commonly used to substitute a default value when a column might be NULL.
<pre><code>SELECT name, COALESCE(phone, 'No phone on file') AS contact FROM customers;</code></pre></p>

<p><strong>Q43. What is a CHECK constraint?</strong><br/>
A CHECK constraint enforces a boolean condition on column values at the database level, rejecting any INSERT/UPDATE that would violate it — independent of application-level validation.
<pre><code>CREATE TABLE products (
  price DECIMAL CHECK (price > 0)
);
-- INSERT INTO products (price) VALUES (-5); would be rejected</code></pre></p>

<p><strong>Q44. What is the difference between UNIQUE constraint and PRIMARY KEY?</strong><br/>
Both enforce uniqueness, but a table can have only one PRIMARY KEY (which also implies NOT NULL), while it can have multiple UNIQUE constraints — and UNIQUE columns can allow one NULL value in most databases.
<pre><code>CREATE TABLE users (
  id INT PRIMARY KEY,
  email VARCHAR UNIQUE   -- unique, but could allow one NULL
);</code></pre></p>

<p><strong>Q45. How would you optimize a slow SQL query?</strong><br/>
Check the EXPLAIN plan for full table scans, add indexes on columns used in WHERE/JOIN/ORDER BY, avoid SELECT * (fetch only needed columns), avoid functions on indexed columns in WHERE (breaks index usage), and rewrite correlated subqueries as JOINs where possible.
<pre><code>-- prevents index usage on order_date:
WHERE YEAR(order_date) = 2026
-- rewrite so the index CAN be used:
WHERE order_date >= '2026-01-01' AND order_date &lt; '2027-01-01'</code></pre></p>

<p><strong>Q46. What is the difference between a natural join and an explicit JOIN...ON?</strong><br/>
NATURAL JOIN automatically joins on all columns with the same name in both tables (implicit, risky if column names coincidentally match or change), while JOIN...ON lets you explicitly specify the join condition — strongly preferred in production for clarity and safety.<br/>
<em>Example:</em> NATURAL JOIN silently breaks if someone adds an unrelated column with a matching name to either table.</p>

<p><strong>Q47. What are window frames (ROWS BETWEEN)?</strong><br/>
A window frame defines which subset of rows within a partition a window function considers, e.g., a running total or moving average — specified with ROWS BETWEEN or RANGE BETWEEN.
<pre><code>SELECT order_date, amount,
       SUM(amount) OVER (ORDER BY order_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_sum_3
FROM orders;
-- computes a 3-row moving sum</code></pre></p>

<p><strong>Q48. What is the difference between LEFT JOIN and LEFT JOIN ... WHERE right.col IS NULL?</strong><br/>
A plain LEFT JOIN returns all left rows plus matches (NULL where unmatched). Adding <code>WHERE right_table.col IS NULL</code> filters down to only left rows with no match at all — a common pattern for finding "records with no related record."
<pre><code>SELECT c.* FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;   -- customers who have never placed an order</code></pre></p>

<p><strong>Q49. What is a materialized view, and how is it different from a regular view?</strong><br/>
A regular view re-runs its underlying query every time it's queried (always fresh, but can be slow). A materialized view physically stores the query result and must be manually/periodically refreshed — much faster to read but can serve stale data until refreshed.
<pre><code>CREATE MATERIALIZED VIEW monthly_sales AS
SELECT DATE_TRUNC('month', order_date) AS month, SUM(amount) FROM orders GROUP BY 1;
REFRESH MATERIALIZED VIEW monthly_sales;</code></pre></p>

<p><strong>Q50. How would you design a schema for a many-to-many relationship (e.g., students and courses)?</strong><br/>
Use a junction/bridge table containing foreign keys to both related tables (plus its own or a composite primary key), since a direct many-to-many relationship can't be represented with a single foreign key column.
<pre><code>CREATE TABLE enrollments (
  student_id INT REFERENCES students(id),
  course_id INT REFERENCES courses(id),
  enrolled_date DATE,
  PRIMARY KEY (student_id, course_id)
);</code></pre></p>
`
  },
];
