-- Medical AI Database Schema
-- SQL Server Database Setup

USE master;
GO

-- Create database
IF NOT EXISTS (SELECT * FROM sys.databases WHERE name = 'medical_ai')
BEGIN
    CREATE DATABASE medical_ai;
END
GO

USE medical_ai;
GO

-- Create tables
CREATE TABLE users (
    id INT IDENTITY(1,1) PRIMARY KEY,
    username NVARCHAR(50) NOT NULL UNIQUE,
    email NVARCHAR(100) NOT NULL UNIQUE,
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE TABLE medical_qa_log (
    id INT IDENTITY(1,1) PRIMARY KEY,
    user_id NVARCHAR(50) NOT NULL,
    question NVARCHAR(MAX) NOT NULL,
    answer NVARCHAR(MAX),
    confidence FLOAT,
    created_at DATETIME2 DEFAULT GETDATE()
);

CREATE TABLE medical_knowledge (
    id INT IDENTITY(1,1) PRIMARY KEY,
    topic NVARCHAR(100) NOT NULL,
    content NVARCHAR(MAX) NOT NULL,
    keywords NVARCHAR(500),
    confidence FLOAT DEFAULT 0.8,
    created_at DATETIME2 DEFAULT GETDATE()
);

-- Insert sample data
INSERT INTO medical_knowledge (topic, content, keywords, confidence) VALUES 
('diabetes', N'อาการของโรคเบาหวาน ได้แก่ กระหายน้ำมาก ปัสสาวะบ่อย น้ำหนักลด เหนื่อยง่าย มองเห็นเบลอ', N'เบาหวาน,diabetes,น้ำตาล,กระหาย,ปัสสาวะ', 0.9),
('hypertension', N'การรักษาความดันโลหิตสูง ควรออกกำลังกาย กินอาหารเค็มน้อย ผักผลไม้มาก และทานยาตามแพทย์สั่ง', N'ความดัน,โลหิตสูง,hypertension,เค็ม,ออกกำลัง', 0.85),
('fever', N'ไข้เป็นอาการที่ร่างกายมีอุณหภูมิสูงกว่าปกติ ควรพักผ่อน ดื่มน้ำเยอะ ถ้าไข้สูงควรไปพบแพทย์', N'ไข้,fever,อุณหภูมิ,ร้อน', 0.8);

GO