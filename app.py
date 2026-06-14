from pathlib import Path
from io import BytesIO
from datetime import datetime
import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =========================
# 基础配置
# =========================

st.set_page_config(
    page_title="全国电网代购价统计",
    page_icon="💰",
    layout="wide"
)

DEFAULT_EXCEL_PATH = Path("data/全国电网代购价统计表.xlsx")


# =========================
# 工具函数
# =========================

def infer_year(sheet_name: str) -> str:
    match = re.search(r"(20\d{2})", sheet_name)
    return match.group(1) if match else "未分类"


def infer_table_type(sheet_name: str) -> str:
    if "首页" in sheet_name:
        return "首页"
    if "总表" in sheet_name:
        return "总表"
    if "分表" in sheet_name:
        return "分表"
    return "其他"


def clean_raw_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")
    df = df.fillna("")
    df.columns = [f"列{i + 1}" for i in range(len(df.columns))]
    return df


def filter_by_keyword(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if not keyword or not keyword.strip():
        return df
    mask = df.astype(str).apply(
        lambda row: row.str.contains(re.escape(keyword.strip()), case=False, na=False).any(),
        axis=1
    )
    return df[mask]


def get_last_modified_text(path: Path) -> str:
    if not path.exists():
        return "未知"
    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# =========================
# Excel 读取（热更新核心）
# 用文件修改时间作为 cache key，
# 文件一更新，st.cache_data 自动失效并重新读取
# =========================

def _get_mtime(path: Path) -> float:
    """获取文件修改时间，作为缓存失效的 key"""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="正在读取 Excel...")
def load_excel_from_path(path: str, _mtime: float):
    """
    _mtime 作为隐藏参数传入，使得文件修改后缓存自动失效。
    参数名以下划线开头，Streamlit 不会对它做哈希，由调用方保证传入正确值。
    """
    return pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")


@st.cache_data(show_spinner="正在读取上传的 Excel...")
def load_excel_from_bytes(file_bytes: bytes):
    return pd.read_excel(BytesIO(file_bytes), sheet_name=None, header=None, engine="openpyxl")


# =========================
# 数据提取辅助函数
# =========================

def _num_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")


def _sum_rows(df: pd.DataFrame, row_idx: list, month_cols: list) -> list:
    if not row_idx:
        return [np.nan] * 12
    data = _num_frame(df.loc[row_idx, month_cols])
    return data.sum(axis=0, min_count=1).reindex(month_cols).tolist()


def _single_row(df: pd.DataFrame, row_idx: list, month_cols: list) -> list:
    if not row_idx:
        return [np.nan] * 12
    data = pd.to_numeric(df.loc[row_idx[0], month_cols], errors="coerce")
    return data.reindex(month_cols).tolist()


def _rows_contains(df, province_col, item_col, province, keywords) -> list:
    sub = df[df[province_col].astype(str).str.strip().eq(str(province).strip())]
    mask = sub[item_col].astype(str).apply(lambda x: any(k in x for k in keywords))
    return sub[mask].index.tolist()


def _row_by_province(df, province_col, province) -> list:
    return df[df[province_col].astype(str).str.strip().eq(str(province).strip())].index.tolist()


def split_voltage_level(voltage_text: str):
    text = str(voltage_text).replace(" ", "").strip()
    for prefix in ["单一制", "两部制"]:
        if text.startswith(prefix):
            return prefix, text.replace(prefix, "", 1)
    return None, text


# =========================
# 业务数据获取函数
# =========================

def get_available_provinces(excel_data) -> list:
    df = excel_data.get("分表5-政府性基金及附加")
    if df is None:
        return []
    vals = df.iloc[1:, 1].dropna().astype(str).str.strip().tolist()
    return list(dict.fromkeys(vals))


def get_voltage_options(excel_data) -> list:
    df = excel_data.get("分表4-电度输配电价")
    if df is None:
        return []
    tmp = df.iloc[1:, [2, 3]].dropna()
    opts = (tmp[2].astype(str).str.strip() + tmp[3].astype(str).str.strip()).tolist()
    return list(dict.fromkeys(opts))


def get_transmission_fee(excel_data, province: str, voltage_level: str):
    df = excel_data.get("分表4-电度输配电价")
    if df is None:
        return np.nan
    billing_type, voltage = split_voltage_level(voltage_level)
    mask = (
        df[1].astype(str).str.strip().eq(str(province).strip())
        & df[2].astype(str).str.strip().eq(str(billing_type).strip())
        & df[3].astype(str).str.strip().eq(str(voltage).strip())
    )
    vals = pd.to_numeric(df.loc[mask, 4], errors="coerce").dropna()
    return float(vals.iloc[0]) if len(vals) else np.nan


def get_gov_fee(excel_data, province: str):
    df = excel_data.get("分表5-政府性基金及附加")
    if df is None:
        return np.nan
    mask = df[1].astype(str).str.strip().eq(str(province).strip())
    vals = pd.to_numeric(df.loc[mask, 3], errors="coerce").dropna()
    return float(vals.iloc[0]) if len(vals) else np.nan


def get_2026_agent_fee(excel_data, province: str) -> list:
    df = excel_data.get("分表1-2026电网代购价")
    if df is None:
        return [np.nan] * 12
    rows = _rows_contains(df, province_col=2, item_col=1, province=province,
                          keywords=["当月平均购电价格", "历史偏差电费折价"])
    return _sum_rows(df, rows, list(range(3, 15)))


def get_2026_line_loss(excel_data, province: str) -> list:
    df = excel_data.get("分表2-2026上网环节线损")
    if df is None:
        return [np.nan] * 12
    rows = _row_by_province(df, province_col=1, province=province)
    return _single_row(df, rows, list(range(2, 14)))


def get_2026_system_fee(excel_data, province: str) -> list:
    df = excel_data.get("分表3-2026系统运行费")
    if df is None:
        return [np.nan] * 12
    province = str(province).strip()
    for c in range(df.shape[1] - 12):
        if str(df.iat[1, c]).strip() == province:
            month_cols = list(range(c + 1, c + 13))
            return _num_frame(df.iloc[2:, month_cols]).sum(axis=0, min_count=1).tolist()
    return [np.nan] * 12


def get_2025_agent_fee(excel_data, province: str) -> list:
    df = excel_data.get("分表8-2025三类细分价")
    if df is None:
        return [np.nan] * 12
    rows = _rows_contains(df, province_col=1, item_col=2, province=province,
                          keywords=["当月平均购电价格", "历史偏差电费折价"])
    return _sum_rows(df, rows, list(range(3, 15)))


def get_2025_line_loss(excel_data, province: str) -> list:
    df = excel_data.get("分表8-2025三类细分价")
    if df is None:
        return [np.nan] * 12
    rows = _rows_contains(df, province_col=1, item_col=2, province=province,
                          keywords=["上网环节线损费用折价"])
    return _single_row(df, rows, list(range(3, 15)))


def get_2025_system_fee(excel_data, province: str) -> list:
    df = excel_data.get("分表8-2025三类细分价")
    if df is None:
        return [np.nan] * 12
    sub = df[df[1].astype(str).str.strip().eq(str(province).strip())]
    system_idx = sub[sub[2].astype(str).str.contains("系统运行费用折合度电水平", na=False)].index.tolist()
    if not system_idx:
        return [np.nan] * 12
    rows = sub[sub.index >= system_idx[0] + 1].index.tolist()
    return _sum_rows(df, rows, list(range(3, 15)))


# =========================
# 图表数据构建
# =========================

def get_parts_for_year(excel_data, year: int, province: str, voltage_level: str) -> pd.DataFrame:
    if year == 2025:
        agent = get_2025_agent_fee(excel_data, province)
        line_loss = get_2025_line_loss(excel_data, province)
        system_fee = get_2025_system_fee(excel_data, province)
    elif year == 2026:
        agent = get_2026_agent_fee(excel_data, province)
        line_loss = get_2026_line_loss(excel_data, province)
        system_fee = get_2026_system_fee(excel_data, province)
    else:
        raise ValueError("只支持 2025 / 2026")

    transmission = [get_transmission_fee(excel_data, province, voltage_level)] * 12
    gov = [get_gov_fee(excel_data, province)] * 12

    parts = pd.DataFrame({
        "代理购电价格": agent,
        "上网环节线损电价": line_loss,
        "电度输配电价": transmission,
        "政府性基金及附加": gov,
        "系统运行费折价": system_fee,
    })

    parts["综合平段电价--电网代购"] = parts.sum(axis=1, min_count=1)
    parts["year"] = year
    parts["month"] = range(1, 13)
    parts["date"] = pd.to_datetime(
        parts["year"].astype(str) + "-" + parts["month"].astype(str) + "-01"
    )
    return parts


@st.cache_data(show_spinner=False)
def build_price_long_cached(
    _excel_data_id: int,   # 用 id() 作为代理 key，让缓存感知数据变化
    province: str,
    voltage_level: str,
    # 真正的数据通过下面这个参数传入，不参与哈希（以下划线开头）
    _excel_data
) -> pd.DataFrame:
    frames = [
        get_parts_for_year(_excel_data, 2025, province, voltage_level),
        get_parts_for_year(_excel_data, 2026, province, voltage_level),
    ]
    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(
        id_vars=["year", "month", "date"],
        value_vars=[
            "代理购电价格", "上网环节线损电价", "电度输配电价",
            "政府性基金及附加", "系统运行费折价", "综合平段电价--电网代购",
        ],
        var_name="指标",
        value_name="电价"
    )
    long["电价"] = pd.to_numeric(long["电价"], errors="coerce")
    return long


def build_price_long(excel_data, province: str, voltage_level: str) -> pd.DataFrame:
    """对外接口，自动传入缓存 key"""
    return build_price_long_cached(
        _excel_data_id=id(excel_data),
        province=province,
        voltage_level=voltage_level,
        _excel_data=excel_data
    )


# =========================
# 图表绘制
# =========================

def draw_line_chart(data: pd.DataFrame, title: str):
    data = data.dropna(subset=["电价"])
    if data.empty:
        st.info(f"{title} 暂无可展示数据")
        return

    chart = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="月份", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("电价:Q", title="元/kWh", scale=alt.Scale(zero=False)),
            color=alt.Color("系列:N", title="系列"),
            tooltip=[
                alt.Tooltip("系列:N", title="系列"),
                alt.Tooltip("date:T", title="月份", format="%Y-%m"),
                alt.Tooltip("电价:Q", title="电价", format=".4f"),
            ]
        )
        .properties(title=title, height=340)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


# =========================
# 页面主体
# =========================

st.title("💰 全国电网代购价统计")
st.caption("数据来源：本地 Excel（data/全国电网代购价统计表.xlsx）。更新 Excel 文件后刷新页面即可看到最新数据。")

with st.sidebar:
    st.header("数据来源")
    uploaded_file = st.file_uploader("临时上传 Excel（覆盖本地文件）", type=["xlsx", "xlsm", "xls"])
    st.divider()
    st.header("筛选条件")


# =========================
# 读取数据（热更新逻辑）
# =========================

excel_data = None
source_label = ""

if uploaded_file is not None:
    excel_data = load_excel_from_bytes(uploaded_file.getvalue())
    source_label = f"📤 上传文件：{uploaded_file.name}"
else:
    if DEFAULT_EXCEL_PATH.exists():
        mtime = _get_mtime(DEFAULT_EXCEL_PATH)
        excel_data = load_excel_from_path(str(DEFAULT_EXCEL_PATH), _mtime=mtime)
        source_label = f"📁 本地文件（最后更新：{get_last_modified_text(DEFAULT_EXCEL_PATH)}）"
    else:
        st.warning("⚠️ 未找到本地 Excel。请将文件放到 `data/全国电网代购价统计表.xlsx`，或在左侧上传。")
        st.stop()

sheet_names = list(excel_data.keys())
sheet_info = pd.DataFrame({
    "sheet_name": sheet_names,
    "year": [infer_year(n) for n in sheet_names],
    "table_type": [infer_table_type(n) for n in sheet_names],
})


# =========================
# 侧边栏筛选
# =========================

with st.sidebar:
    years = ["全部"] + sorted(
        [y for y in sheet_info["year"].unique() if y != "未分类"], reverse=True
    )
    if "未分类" in sheet_info["year"].values:
        years.append("未分类")

    selected_year = st.selectbox("选择年份", years)
    selected_type = st.selectbox("选择表类型", ["全部", "总表", "分表", "其他"])

    filtered = sheet_info.copy()
    if selected_year != "全部":
        filtered = filtered[filtered["year"] == selected_year]
    if selected_type != "全部":
        filtered = filtered[filtered["table_type"] == selected_type]

    available_sheets = filtered["sheet_name"].tolist()
    if not available_sheets:
        st.warning("没有符合条件的工作表")
        st.stop()

    selected_sheet = st.selectbox("选择工作表", available_sheets)
    keyword = st.text_input("关键词搜索", placeholder="例如：广东、江苏、尖峰")

    st.divider()
    st.header("📈 图表设置")

    province_options = get_available_provinces(excel_data)
    voltage_options = get_voltage_options(excel_data)

    if not province_options:
        st.warning("未识别到省份列表(需要分表5-政府性基金及附加)")
        st.stop()
    if not voltage_options:
        st.warning("未识别到电压等级列表(需要分表4-电度输配电价)")
        st.stop()

    default_province = "江苏" if "江苏" in province_options else province_options[0]
    default_voltage = "两部制220kV" if "两部制220kV" in voltage_options else voltage_options[0]

    chart_province = st.selectbox("图表省份", province_options,
                                   index=province_options.index(default_province))
    chart_voltage = st.selectbox("图表电压等级", voltage_options,
                                  index=voltage_options.index(default_voltage))

    default_compare = [p for p in ["江苏", "浙江", "上海", "安徽", "山东"] if p in province_options]
    if not default_compare:
        default_compare = province_options[:5]

    compare_provinces = st.multiselect("省间对比省份", province_options, default=default_compare)


# =========================
# 顶部信息栏
# =========================

col1, col2, col3, col4 = st.columns(4)

raw_df = excel_data[selected_sheet]
df = clean_raw_dataframe(raw_df)
df = filter_by_keyword(df, keyword)

with col1:
    st.metric("数据来源", "本地文件" if uploaded_file is None else "上传文件")
with col2:
    st.metric("当前工作表", selected_sheet)
with col3:
    st.metric("行数", len(df))
with col4:
    st.metric("列数", len(df.columns))

st.caption(source_label)


# =========================
# 动态图表
# =========================

st.divider()
st.header("📈 动态图表")

try:
    chart_data = build_price_long(excel_data, chart_province, chart_voltage)

    tab1, tab2, tab3 = st.tabs(["到户电价", "浮动价格", "省间电价对比"])

    with tab1:
        d = chart_data[chart_data["指标"].isin(["代理购电价格", "综合平段电价--电网代购"])].copy()
        d["系列"] = d["指标"]
        draw_line_chart(d, f"{chart_province}｜{chart_voltage}｜到户电价")
        st.dataframe(
            d.pivot_table(index="date", columns="系列", values="电价", aggfunc="first").reset_index(),
            use_container_width=True, hide_index=True
        )

    with tab2:
        d = chart_data[chart_data["指标"].isin(["上网环节线损电价", "系统运行费折价"])].copy()
        d["系列"] = d["指标"]
        draw_line_chart(d, f"{chart_province}｜{chart_voltage}｜浮动价格")
        st.dataframe(
            d.pivot_table(index="date", columns="系列", values="电价", aggfunc="first").reset_index(),
            use_container_width=True, hide_index=True
        )

    with tab3:
        frames = []
        for p in compare_provinces:
            one = build_price_long(excel_data, p, chart_voltage)
            one = one[one["指标"].eq("综合平段电价--电网代购")].copy()
            one["系列"] = p
            frames.append(one)

        if frames:
            compare_data = pd.concat(frames, ignore_index=True)
            draw_line_chart(compare_data, f"{chart_voltage}｜省间电价对比")
            st.dataframe(
                compare_data.pivot_table(index="date", columns="系列", values="电价", aggfunc="first").reset_index(),
                use_container_width=True, hide_index=True
            )
        else:
            st.info("请选择至少一个省份进行对比")

except Exception as e:
    st.error("图表生成失败，请检查 Excel 分表结构是否发生变化。")
    st.exception(e)


# =========================
# 原始数据表
# =========================

st.divider()
st.subheader(f"📄 {selected_sheet}")
st.dataframe(df, use_container_width=True, hide_index=True)
st.download_button(
    label="⬇️ 下载当前表为 CSV",
    data=dataframe_to_csv_bytes(df),
    file_name=f"{selected_sheet}.csv",
    mime="text/csv"
)

# =========================
# Sheet 总览
# =========================

with st.expander("查看全部工作表结构"):
    st.dataframe(sheet_info, use_container_width=True, hide_index=True)