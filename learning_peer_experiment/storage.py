import sqlite3


def replace_records(path, rows):
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS observations (period TEXT, group_name TEXT, count INTEGER, value REAL)")
        connection.execute("DELETE FROM observations")
        connection.executemany("INSERT INTO observations VALUES (?, ?, ?, ?)",
                               [(row["period"], row["group"], int(row["count"]), float(row["value"]))
                                for row in rows])
        return connection.execute("SELECT period, group_name, count, value FROM observations ORDER BY period, group_name").fetchall()


def replace_trial_records(path, trial):
    """把试验登记记录写入 SQLite，派生分析只从这些原始登记出发，原始 CSV 不被修改。"""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS design (
                period TEXT, group_name TEXT, arm TEXT, arm_label TEXT,
                invite_flag INTEGER, probability REAL, invite_fraction REAL);
            CREATE TABLE IF NOT EXISTS persons (person_id TEXT PRIMARY KEY, group_name TEXT, eligible INTEGER);
            CREATE TABLE IF NOT EXISTS assignments (
                period TEXT, person_id TEXT, group_name TEXT, assigned_arm TEXT, assigned_invite INTEGER);
            CREATE TABLE IF NOT EXISTS notifications (
                period TEXT, person_id TEXT, group_name TEXT, notified INTEGER);
            CREATE TABLE IF NOT EXISTS attendance (
                period TEXT, person_id TEXT, group_name TEXT, attended INTEGER);
            CREATE TABLE IF NOT EXISTS edges (
                period TEXT, from_person TEXT, to_person TEXT, pre_treatment INTEGER);
            CREATE TABLE IF NOT EXISTS budget (period TEXT, invitations_budget INTEGER);
            """)
        for table in ("design", "persons", "assignments", "notifications",
                      "attendance", "edges", "budget"):
            connection.execute(f"DELETE FROM {table}")
        connection.executemany("INSERT INTO design VALUES (?, ?, ?, ?, ?, ?, ?)",
                               [(r.period, r.group, r.arm, r.arm_label, r.invite_flag,
                                 r.probability, r.invite_fraction) for r in trial.design])
        connection.executemany("INSERT INTO persons VALUES (?, ?, ?)",
                               [(pid, g, 1) for pid, g in trial.persons])
        connection.executemany("INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                               [(r.period, r.person_id, r.group, r.assigned_arm, r.assigned_invite)
                                for r in trial.assignments])
        connection.executemany("INSERT INTO notifications VALUES (?, ?, ?, ?)",
                               [(r.period, r.person_id, r.group, r.notified)
                                for r in trial.notifications])
        connection.executemany("INSERT INTO attendance VALUES (?, ?, ?, ?)",
                               [(r.period, r.person_id, r.group, r.attended)
                                for r in trial.attendance])
        connection.executemany("INSERT INTO edges VALUES (?, ?, ?, ?)",
                               [(r.period, r.from_person, r.to_person, r.pre_treatment)
                                for r in trial.edges])
        connection.executemany("INSERT INTO budget VALUES (?, ?)",
                               [(r.period, r.invitations_budget) for r in trial.budget])
        return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("design", "persons", "assignments", "notifications",
                              "attendance", "edges", "budget")}
