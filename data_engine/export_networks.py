# -*- coding: utf-8 -*-
"""
export_networks.py — 将实验所需的全部标准配电网导出为可移植的 pandapower JSON。
在实验机上可直接 pp.from_json() 加载，无需再依赖 simbench/pandapower 版本。

输出目录: 01_数据/网络模型与场景真值/networks/
每张网络附带一个 _meta.json（来源、电压等级、规模）。
"""
import os
import json
import pandapower as pp
import pandapower.networks as nw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "01_数据", "网络模型与场景真值", "networks")
os.makedirs(OUT, exist_ok=True)


def export(net, name, source, role):
    # 统一命名并保存
    net.name = name
    f = os.path.join(OUT, f"{name}.json")
    pp.to_json(net, f)
    meta = {
        "name": name,
        "source": source,
        "role": role,  # train-ID / eval-OOD 等
        "n_bus": len(net.bus),
        "n_line": len(net.line),
        "n_trafo": len(net.trafo),
        "n_load": len(net.load),
        "n_sgen": len(net.sgen),
        "vn_kv_min": float(net.bus.vn_kv.min()),
        "vn_kv_max": float(net.bus.vn_kv.max()),
    }
    with open(os.path.join(OUT, f"{name}_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"[OK] {name:28s} bus={meta['n_bus']:4d} line={meta['n_line']:4d} "
          f"load={meta['n_load']:4d} sgen={meta['n_sgen']:3d} ({role})")


def main():
    # ---------- 训练域内（ID）网络 ----------
    export(nw.case33bw(), "IEEE33", "pandapower case33bw (Baran & Wu 1989)", "train-ID + 主战场")
    export(nw.create_cigre_network_mv(with_der=False), "CIGRE_MV", "CIGRE TB-608 WG C6.4.02 MV", "train-ID")
    # 注: IEEE69-bus 不在当前 pandapower 内置集中，如需可按 Baran & Wu (1989) 发表参数表手工构建
    # （标准做法：12.66kV 辐射网 + 5 条常开联络线，总负荷 3802.2 kW / 2694.6 kVAr）

    # ---------- 泛化（OOD）测试网络：只测不训 ----------
    export(nw.case89pegase(), "PEGASE89", "PEGASE 89-bus (transmission-style)", "eval-OOD")
    export(nw.case118(), "IEEE118", "pandapower case118 (transmission-style)", "eval-OOD")
    export(nw.case145(), "IEEE145", "pandapower case145", "eval-OOD")
    export(nw.case_illinois200(), "Illinois200", "UIUC 200-bus", "eval-OOD")

    # ---------- SimBench 德国真实风格网络（自带时序 profile） ----------
    import simbench as sb
    for code, role in [
        ("1-MV-urban--0-sw", "train-ID(SimBench域)"),
        ("1-MV-semiurb--0-sw", "eval-OOD(SimBench域)"),
        ("1-MV-rural--0-sw", "eval-OOD(SimBench域)"),
    ]:
        net = sb.get_simbench_net(code)
        export(net, f"SimBench_{code}", f"SimBench {code} (Meinecke et al. 2020)", role)

    print("\n全部网络导出完成 →", os.path.abspath(OUT))


if __name__ == "__main__":
    main()
