/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import * as React from 'react'
import { useResizeEventDependency } from '../../utils/resize'
import * as echarts from 'echarts'

interface IProps {
  title?: string
  units?: string
  colors?: Array<string>
  chartData: ColumnChartData
}

export interface ColumnChartData {
  legends: Array<string>
  barLabels: Array<string>
  barHeights: Array<Array<number>>
}

export const ColumnChart: React.FC<IProps> = (props) => {
  const { title, units, colors, chartData } = props
  const { legends, barLabels, barHeights } = chartData
  const graphRef = React.useRef<HTMLDivElement>(null)
  const [resizeEventDependency] = useResizeEventDependency()

  const getAngleByDataLength = (data: number) => {
    if (data < 10) {
      return 0
    } else {
      // 数量越大越趋近于旋转90度
      return 90 * (1 - 10 / data)
    }
  }

  React.useLayoutEffect(() => {
    const element = graphRef.current
    if (!element) return

    const chart = echarts.init(element)
    const dataSource: Array<Array<number | string>> = []
    dataSource.push(['worker', ...legends])
    barHeights.forEach((item, index) => {
      barLabels[index] !== undefined && dataSource.push([barLabels[index], ...item])
    })
    const options: echarts.EChartsOption = {
      title: {
        text: title
      },
      legend: {
        bottom: 0
      },
      xAxis: {
        type: 'category',
        axisLabel: {
          interval: 0,
          rotate: getAngleByDataLength(barLabels.length),
          formatter: (name: string) => {
            const index = name.indexOf('@')
            if (index > -1) {
              name = name.slice(index + 1)
            }
            return name.length > 16 ? name.slice(0, 14) + "..." : name;
          }
        }
      },
      yAxis: {
        type: 'value',
        name: units,
        nameTextStyle: {
          fontSize: 16
        }
      },
      tooltip: {
        trigger: 'item'
      },
      dataset: {
        source: dataSource
      },
      series: Array(legends.length).fill({
        type: 'bar',
        stack: 'samesign'
      }),
    }
    if (colors) {
      options.color = colors.slice(0, barLabels.length)
    }

    options && chart.setOption(options, true)
    return () => {
      chart.dispose()
    }
  }, [title, chartData, resizeEventDependency])

  return (
    <div ref={graphRef} style={{ height: '500px' }}></div>
  )
}
