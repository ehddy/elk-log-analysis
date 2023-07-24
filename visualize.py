import plotly.subplots as sp
import plotly.figure_factory as ff
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.subplots as sp
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')


class Visualize:        
    ### 시각화 함수 모음 
    def category_pie_plot(self, data, column_name):

        res_counts = data[column_name].value_counts()[:5]

        # 파이 차트 생성
        fig = go.Figure(data=go.Pie(labels=res_counts.index, values=res_counts.values))

        fig.update_layout(title=f'{column_name} Distribution')
        
        return fig

        
        
    def minute_connect_plot(self, data):
        # 데이터를 시간적으로 그룹화하여 그룹별 통계 계산
        data['date_group_minute'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        grouped_data = data.groupby('date_group_minute')['date_group_minute'].count()

        

        hour_name = data.index[0].strftime("%Y년 %m월 %d일 %H시 기준")
        user_name = data['sDevID'][0]

        # 선 그래프 그리기
        fig = px.line(grouped_data, x=grouped_data.index, y=grouped_data.values, title='Access Pattern(ID = {}) : {}'.format(user_name,hour_name))
        
        return fig


        
    def histogram(self, data, column_name):
        # 데이터 프레임에서 '접속 UA 수' 열을 추출하여 histogram 그래프 생성
        fig = px.histogram(data, x=column_name,  title=f'{column_name} histogram')

        # 그래프 출력
        
        return fig

        
    def minute_connect_category_plot(self, segment_data, column_name):
        # 차트를 그릴 데이터를 담을 빈 리스트 생성
        plot_data = []

        # 각 데이터프레임에 대해 그래프를 그리기 위한 데이터를 생성하여 plot_data 리스트에 추가
        for df in segment_data:
            df['date_group_minute'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            grouped_data = df.groupby('date_group_minute')['date_group_minute'].count()
            plot_data.append(grouped_data)

        # 차트 그리기
        fig = go.Figure()

        for i, data in enumerate(plot_data):
            hour_name = segment_data[i].index[0].strftime("%Y년 %m월 %d일 %H시 기준")
            ua_name = segment_data[i][column_name][0]

            fig.add_trace(go.Scatter(x=data.index, y=data.values, name='{}'.format(ua_name)))

        fig.update_layout(title='Access Patterns User ID : {}'.format(segment_data[0]['sDevID'][0]), xaxis_title='Minute', yaxis_title='Count', 
                        width=1200,  # 그래프의 가로 크기 (픽셀 단위)
                        height=600# 그래프의 세로 크기 (픽셀 단위)
                    )
        
        return fig

        
    def minute_duration_plot(self, data):
        data['date_group_minute'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        # 데이터를 시간적으로 그룹화하여 그룹별 통계 계산
        grouped_data = data.groupby('date_group_minute')['duration'].mean()

        hour_name = data.index[0].strftime("%Y년 %m월 %d일 %H시 기준")
        user_name = data['sDevID'][0]

        # 선 그래프 그리기
        fig = px.line(grouped_data, x=grouped_data.index, y=grouped_data.values, title='Duration Pattern(ID = {}) : {}'.format(user_name,hour_name))
        
        return fig

        
        
    def minute_duration_category_plot(self, segment_data, column_name):
        # 차트를 그릴 데이터를 담을 빈 리스트 생성
        plot_data = []
        # 각 데이터프레임에 대해 그래프를 그리기 위한 데이터를 생성하여 plot_data 리스트에 추가
        for df in segment_data:
            df['date_group_minute'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            grouped_data = df.groupby('date_group_minute')['duration'].mean()
            plot_data.append(grouped_data)

        # 차트 그리기
        fig = go.Figure()

        for i, data in enumerate(plot_data):
            hour_name = segment_data[i].index[0].strftime("%Y년 %m월 %d일 %H시 기준")
            category_name = segment_data[i][column_name][0]

            fig.add_trace(go.Scatter(x=data.index, y=data.values, name='{}'.format(category_name)))

        fig.update_layout(title='Duration Patterns User ID : {}'.format(segment_data[0]['sDevID'][0]), xaxis_title='Minute', yaxis_title='Mean', 
                        width=1200,  # 그래프의 가로 크기 (픽셀 단위)
                        height=600# 그래프의 세로 크기 (픽셀 단위)
                    )
        
        return fig
    

    
    def scatter_plot(self, data, column_1, column_2):
        # Scatter plot 생성
        fig = px.scatter(data, x=column_1, y=column_2, title=f'{column_1} X {column_2}', template='none')
        fig.update_layout(
            xaxis_title=column_1,
            yaxis_title=column_2, 
            width=800,  # 그래프의 가로 크기 (픽셀 단위)
            height=800)# 그래프의 세로 크기 (픽셀 단위)
        return fig

    def scatter_plot_label(self, data, column_1, column_2, label):
        # Scatter plot 생성
        fig = px.scatter(data, x=column_1, y=column_2,  color=data[label].values ,title=f'{column_1} X {column_2}',template='none')
        fig.update_layout(
            xaxis_title=column_1,
            yaxis_title=column_2, 
            width=800,  # 그래프의 가로 크기 (픽셀 단위)
            height=800)# 그래프의 세로 크기 (픽셀 단위)
        return fig
    
    
    def scatter_plot_3d(self, data, column_1, column_2, column_3):
        # Scatter plot 생성
        fig = px.scatter_3d(data, x=column_1, y=column_2, z=column_3,title=f'{column_1} X {column_2}',template='none')

        return fig
    
    def scatter_plot_3d_label(self, data, column_1, column_2, column_3, label):
        # Scatter plot 생성
        fig = px.scatter_3d(data, x=column_1, y=column_2, z=column_3, color=data[label].values,title=f'{column_1} X {column_2}',template='none')

        return fig
    

    def describe_bar_chart(self, data, column, label):
        
        # # 막대 차트 그리기
        fig = go.Figure(data=[go.Bar(x=data[label], y=data[column])])

        fig.update_layout(
            title=f'{label}별 가입자의 {column}',
            xaxis_title=f'{label}',
            yaxis_title=f'{column}',
            width=800,  # 그래프의 가로 크기 (픽셀 단위)
            height=1000, # 그래프의 세로 크기 (픽셀 단위)
            xaxis_tickangle=-45  # 라벨 이름을 오른쪽으로 회전시킴

        )

        return fig

    def value_counts_top10_bar(self, data, column_name):
        # 데이터셋의 'sHost' 열 값 빈도수 계산
        top_hosts = data[column_name].value_counts()[:10]
        


        
        fig = go.Figure(data=[go.Bar(x=top_hosts.index, y=top_hosts.values)])

        # 차트 레이아웃 설정
        fig.update_layout(
            title=f"Top 10 {column_name}",
            xaxis_title="Frequency",
            yaxis_title=column_name
        )

        # 차트 출력
        return fig
        
        
    def value_counts_top10_bar_reverse(self, data, column_name):
        # 데이터셋의 'sHost' 열 값 빈도수 계산
        top_hosts = data[column_name].value_counts()[:10]
        # top_hosts = top_hosts.index.str.slice(0, 20)
        # 시리즈 객체 순서 반전
        top_hosts = top_hosts[::-1]

        # 가로 바 차트 생성
        fig = go.Figure(data=[go.Bar(y=[host[:20] + ' ' for host in top_hosts.index], x=top_hosts.values, orientation='h')])

        # 차트 레이아웃 설정
        fig.update_layout(
            title=f"Top 10 {column_name}",
            xaxis_title="Frequency",
            yaxis_title=column_name
        )

        # 차트 출력
        
        return fig

    def heatmap(self, data):
        # 데이터 프레임에서 상관계수 계산

        numerical_feats = list(data.dtypes[data.dtypes != "object"].index)
        correlation_matrix = data[numerical_feats].corr().round(2)


        # 상관계수 히트맵 생성
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values,
            x=list(correlation_matrix.columns),
            y=list(correlation_matrix.index),
            colorscale = "Rainbow",
            showscale=True
        )

        # 히트맵 레이아웃 설정
        fig.update_layout(
            title='Correlation Heatmap',
            width=900,  # 그래프의 너비 조정
            height=900,  # 그래프의 높이 조정
            xaxis=dict(tickfont=dict(size=8)),  # x축 레이블 글꼴 크기 조정
            yaxis=dict(tickfont=dict(size=8)),  # y축 레이블 글꼴 크기 조정

        )


        # 히트맵 출력
        
        return fig


    def groupby_mean_plot(self, data, group_name, mean_column_name):
        group_data = data.groupby(group_name)[mean_column_name].mean().sort_values(ascending=False)[:10]
        
        # 시리즈 객체 순서 반전
        group_data = group_data[::-1]

        
        # 막대 차트 그리기
        fig = go.Figure(data=[go.Bar(y=group_data.index, x=group_data.values,  orientation='h')])

        fig.update_layout(
            title=f'{group_name}별 가입자의 {mean_column_name}',
            xaxis_title=f'{group_name}',
            yaxis_title=f'{mean_column_name}',
            width=1100,  # 그래프의 가로 크기 (픽셀 단위)
            height=800, # 그래프의 세로 크기 (픽셀 단위)
            xaxis_tickangle=-45  # 라벨 이름을 오른쪽으로 회전시킴

        )
        
        return fig 


    def seasonal_decompose_plot(self, data, period=5):

        data['date_group_minute'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        # 데이터를 시간적으로 그룹화하여 그룹별 통계 계산
        grouped_data = data.groupby('date_group_minute')[['sHost']].count()

        result = sm.tsa.seasonal_decompose(grouped_data['sHost'], model='additive', period=period)

        # 시각화용 데이터 생성
        dates = pd.to_datetime(grouped_data.index)
        original_values = grouped_data['sHost'].values
        trend_values = result.trend.values
        seasonal_values = result.seasonal.values
        residual_values = result.resid.values

        # subplot 생성
        fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True)

        # 그래프 추가
        fig.add_trace(go.Scatter(x=dates, y=original_values, name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates, y=trend_values, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=dates, y=seasonal_values, name='Seasonality'), row=3, col=1)
        fig.add_trace(go.Scatter(x=dates, y=residual_values, name='Residuals'), row=4, col=1)

        # 레이아웃 설정
        fig.update_layout(
            title='Seasonal Decomposition',
            height=800,
            width=800
        )

        # subplot 축 레이블 설정 
        fig.update_xaxes(title_text='Date', row=4, col=1)
        fig.update_yaxes(title_text='Original', row=1, col=1)
        fig.update_yaxes(title_text='Trend', row=2, col=1)
        fig.update_yaxes(title_text='Seasonality', row=3, col=1)
        fig.update_yaxes(title_text='Residuals', row=4, col=1)

        # 그래프 출력
        return fig

        
    def gage_chart(self, describe_data, column_name):
        # 필요한 데이터
        value = describe_data[column_name][0]
        min_value = 0
        max_value = 100

        # 게이지 차트 생성
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = column_name,
            gauge = {
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_value, max_value], 'color': 'lightgray'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        return fig


    def text_chart(self, describe_data, column_name):
        # 데이터 프레임에서 특정 값 추출
        value = describe_data[column_name].values[0]

        # 플롯 생성
        fig = go.Figure()

        if column_name == '전체 접속 횟수':
            title = '접속 수(' + describe_data['접속 기간'][0] + ')'
        else:
            title = column_name

        # 특정 값 표시
        fig.add_trace(go.Indicator(
        mode="number",
        value=value,
        title=title,
        ))

        
        # 대시보드 레이아웃 설정
        fig.update_layout(
        title="",
        title_font=dict(size=24),
        )
        
        
        
        # 대시보드 출력
        return fig


    # 데이터프레임에서 데이터 추출 및 뒤집기
    def ip_location_table(self, ip_describe):
        data = ip_describe.transpose().values.tolist()
        columns = ip_describe.columns.tolist()

        # 데이터 표 그림 생성
        fig = go.Figure(data=[go.Table(
            header=dict(values=columns,
                        fill_color='gray',
                        align='center',
                        font=dict(color='black', size=14),
                        height=30),
            cells=dict(values=data,
                    fill=dict(color='white'),
                    align='center',
                    font=dict(color='black', size=12),
                    height=25),
        )])

        # 셀 너비 조정
        fig.update_layout(width=800)

        return fig


    def get_dash_gage(self, describe_data):  
        user_name = describe_data['가입자 ID'][0]
        
        specs = [
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ]
        
        font_family = 'Pretendard Black, sans-serif'



        # {'type': 'xy', 'colspan': 2}, None,
        fig1 = self.gage_chart(describe_data, '차단율(%)')
        fig2 = self.gage_chart(describe_data, '최대 빈도 URL 접속 비율(%)')
        fig3 = self.gage_chart(describe_data, '최다 이용 UA 접속 비율(%)')
        fig4 = self.gage_chart(describe_data, '접속 횟수 대비 고유 URL 비율(%)')
        fig6 = self.text_chart(describe_data, '전체 접속 횟수')
        fig7 = self.text_chart(describe_data, '평균 접속 수(1분)')
        fig8 = self.text_chart(describe_data, '평균 접속 간격(초)')
        fig9 = self.text_chart(describe_data, '접속 UA 수')


        # 대시보드 그래프 배열
        fig = make_subplots(
            rows=2, cols=4,
            vertical_spacing=0.2,
            horizontal_spacing=0.2, 
            specs=specs,  # 그래프 간의 수직 간격 조정
           
        )


        # 그래프에 폰트 적용
        fig.update_layout(
            font=dict(family=font_family)
        )

        #   # Reduce the graph size
        # fig.update_layout(
        #     height=1000,  # Set the height of the entire layout
        #     width=1000,   # Set the width of the entire layout
        # )

        # Reduce the text size
        fig.update_traces(
            textfont=dict(size=10),  # Adjust the font size of the text on the graphs
        )

        fig.add_trace(fig1.data[0], row=2, col=1)
        fig.add_trace(fig2.data[0], row=2, col=2)
        fig.add_trace(fig3.data[0], row=2, col=3)
        fig.add_trace(fig4.data[0], row=2, col=4)
        fig.add_trace(fig6.data[0], row=1, col=1)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig7.data[0], row=1, col=2)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig8.data[0], row=1, col=3)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig9.data[0], row=1, col=4)  # colspan을 사용하여 두 개의 열 차지


        # # 표 레이아웃 설정
        # fig.update_layout(
        #     title={
        #         'text': f"<{user_name}> Status in the Last 1 Hour",
        #         'font': {'size': 20, 'family': font_family}
        #     },
        # )

        return fig 


    def get_dash_ipchart(self, data):
        specs = [
            [{'type': 'xy', 'colspan': 2, 'rowspan': 2}, None, {'type': 'xy', 'colspan': 2, 'rowspan': 2}, None],
            [None, None, None, None],
        ]
        
        font_family = 'Pretendard Black, sans-serif'

        fig10 = self.value_counts_top10_bar(data, 'sHost')
        fig11 = self.value_counts_top10_bar(data, 'uDstIp')

        # 대시보드 그래프 배열
        fig = make_subplots(
            rows=2, cols=4,
            vertical_spacing=0.1,
            horizontal_spacing=0.1, 
            specs=specs,  # 그래프 간의 수직 간격 조정
            subplot_titles=['Top 10 URL', 'Top 10 IP Address']
        )

        fig10.data[0]['showlegend'] = False
        fig11.data[0]['showlegend'] = False


        fig10.data[0]['marker']['color'] = '#7b68ee'
        fig11.data[0]['marker']['color'] = '#7b68ee'

        # 그래프에 폰트 적용
        fig.update_layout(
            font=dict(family=font_family)
        )

      # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig10.data[0], row=1, col=1)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig11.data[0], row=1, col=3)  # colspan을 사용하여 두 개의 열 차지

        fig.update_layout(template='none')

        return fig 

    def get_dash_seasonal(self, data):
        specs = [
            [{'type': 'xy', 'colspan': 4, 'rowspan': 2}, None, None, None],
            [None, None, None, None],
            [{'type': 'xy', 'colspan': 4, 'rowspan': 2}, None, None, None],
            [None, None, None, None]
        ]
        
        font_family = 'Pretendard Black, sans-serif'



        fig12 = self.seasonal_decompose_plot(data, period=5)
  

        # 대시보드 그래프 배열
        fig = make_subplots(
            rows=4, cols=4,
            vertical_spacing=0.2,
            horizontal_spacing=0.2, 
            specs=specs,  # 그래프 간의 수직 간격 조정
            subplot_titles=['Connect Pattern', '', '', '',]        )

        fig12.data[0]['showlegend'] = False
        fig12.data[2]['showlegend'] = False

        fig12.data[0]['marker']['color'] = '#ff009f'
        fig12.data[2]['marker']['color'] = '#dda0dd'

        # 그래프에 폰트 적용
        fig.update_layout(
            font=dict(family=font_family)
        )
         # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig12.data[0], row=1, col=1)
        fig.add_trace(fig12.data[2], row=3, col=1)

        # fig.add_trace(fig12.data[1], row=6, col=1)
        # fig.add_trace(fig12.data[2], row=7, col=1)
        # fig.add_trace(fig12.data[3], row=8, col=1)


        # 특정 그래프의 Y 축 제목 설정
        fig.update_yaxes(title_text='Original', title_font=dict(size=15), row=1, col=1)
        fig.update_yaxes(title_text='Seasonality', title_font=dict(size=15), row=3, col=1)


        fig.update_layout(template='none')

        return fig


    def get_dash_ua_port(self, data):
        specs = [
        [{'type': 'pie', 'colspan': 2, 'rowspan': 4}, None, {'type': 'pie', 'colspan': 2, 'rowspan': 4}, None],
        [None, None, None, None],
        [None, None, None, None], 
        [None, None, None, None]
    ]
            
        font_family = 'Pretendard Black, sans-serif'


        fig13 = self.category_pie_plot(data, 'uDstPort')
        fig11 = self.category_pie_plot(data, 'sUA')
      

        # 대시보드 그래프 배열
        fig = make_subplots(
            rows=4, cols=4,
            # vertical_spacing=0.1,
            # horizontal_spacing=0.1, 
            specs=specs,  # 그래프 간의 수직 간격 조정
            subplot_titles=['UA Ratio', 'Port Number Ratio']        )

        fig11.data[0]['showlegend'] = False
        # fig13.data[0]['showlegend'] = False
     
        fig11.data[0]['marker']['colors'] =  ['#7b68ee', 'd8bfd8', 'b0e0e6', 'ff009f', 'ffc0cb']

        fig13.data[0]['marker']['colors'] =  ['ffc0cb', 'ff009f', 'b0e0e6', 'd8bfd8','#7b68ee']

        # 그래프에 폰트 적용
        fig.update_layout(
            font=dict(family=font_family)
        )

    

        fig.add_trace(fig11.data[0], row=1, col=1)  # colspan을 사용하여 두 개의 열 차지
        fig.add_trace(fig13.data[0], row=1, col=3)  # colspan을 사용하여 두 개의 열 차지


        # fig.add_trace(fig12.data[1], row=6, col=1)
        # fig.add_trace(fig12.data[2], row=7, col=1)
        # fig.add_trace(fig12.data[3], row=8, col=1)


        fig.update_layout(template='none')

        return fig